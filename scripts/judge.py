"""
Judge. Reads both corpora plus its own history, then for each gap:
  1. RETRIEVAL: cosine-similarity shortlist of top-K unseen capabilities
     (this is what avoids O(n*m) brute force as the corpora grow).
  2. BLIND VERDICT: a fit score, computed with no knowledge that
     elaboration exists -- this is what keeps the score from drifting
     toward "yes" just because "yes" leads to a more interesting answer.
  3. GATE: only scores >= THRESHOLD proceed.
  4. PRECEDENT RETRIEVAL + ELABORATION: past similar verdicts are folded
     into the prompt as context -- this is the "gains wisdom" mechanism,
     and it is retrieval, not training; nothing here changes any model
     weights.
  5. AUDIT: entailment check, existence check (search-verified -- if it
     already exists, stop here), perturbation check, adversarial
     cross-exam by a call with zero investment in the idea.
Every already-judged (gap_id, capability_id) pair is skipped permanently.
"""
import copy
from common import (
    load_jsonl, append_jsonl, gemini_generate, gemini_embed, cosine_sim,
    extract_json, new_id, now_iso,
)

GAPS_PATH = "data/gaps.jsonl"
CAPS_PATH = "data/capabilities.jsonl"
VERDICTS_PATH = "data/pair_verdicts.jsonl"

TOP_K = 3
THRESHOLD = 70


def gap_public_fields(g):
    sp = g["structural_properties"]
    return f"""SITUATION 1 (artifact with a gap):
Artifact type: {g['artifact_type']}
Structural properties: cardinality/scale={sp.get('cardinality_scale')}, dimensionality={sp.get('dimensionality')}, production method={sp.get('production_method')}, mutability={sp.get('mutability')}
Complaint: {g['complaint']}
Current workaround: {g.get('workaround', 'none stated')}"""


def cap_public_fields(c):
    co = c["core_operations"]
    return f"""SITUATION 2 (existing capability):
Operates on: {c['operates_on']}
Core operations: selection={co.get('selection')}, transformation={co.get('transformation')}, deletion/reduction={co.get('deletion_reduction')}, I/O={co.get('io')}, delivery environment={co.get('delivery_environment')}
Why cheap to extend: {c['cheap_to_extend_reason']}"""


def find_candidates(gaps, caps, seen_pairs):
    candidates = []
    for g in gaps:
        scored = [
            (cosine_sim(g["embedding"], c["embedding"]), c)
            for c in caps
            if (g["id"], c["id"]) not in seen_pairs
        ]
        scored.sort(key=lambda x: -x[0])
        for sim, c in scored[:TOP_K]:
            candidates.append((g, c, sim))
    return candidates


def blind_verdict(gap, cap):
    prompt = f"""{gap_public_fields(gap)}

{cap_public_fields(cap)}

Score, from 0 to 100, how strongly Situation 2's capability
structurally satisfies Situation 1's requirements. Most pairs presented
this way will NOT be a strong structural fit -- a well-calibrated judge
should give a low score to a clear majority of pairs. Respond with ONLY
a JSON object: {{"fit_score": <0-100 integer>, "justification": "<max
20 words>"}}. No markdown fences."""
    return extract_json(gemini_generate(prompt))


def find_precedents(gap, cap, verdicts, k=3):
    try:
        query_emb = gemini_embed(
            gap_public_fields(gap) + "\n" + cap_public_fields(cap),
            task_type="RETRIEVAL_QUERY",
        )
    except Exception:
        return []
    scored = [
        (cosine_sim(query_emb, v["pair_embedding"]), v)
        for v in verdicts
        if "pair_embedding" in v
    ]
    scored.sort(key=lambda x: -x[0])
    return [v for _, v in scored[:k]]


def elaborate(gap, cap, precedents):
    precedent_text = ""
    if precedents:
        lines = [
            f"- score={p.get('fit_score')}, verdict={p.get('audit_verdict', 'n/a')}, "
            f"bridge={p.get('elaboration', {}).get('stage_4', '')[:200]}"
            for p in precedents
        ]
        precedent_text = "Relevant past judged pairs (context only, do not copy):\n" + "\n".join(lines)

    prompt = f"""{gap_public_fields(gap)}

{cap_public_fields(cap)}

{precedent_text}

Assume these two structurally connect. Answer as a JSON object with
keys stage_1, stage_2, stage_3, stage_4 (each a string):
STAGE 1 -- Restate Situation 1's requirements as a list of operations.
STAGE 2 -- Restate Situation 2's capability as a list of operations.
STAGE 3 -- Map the overlap explicitly; every item must trace to
something stated in Stage 1 or Stage 2 above.
STAGE 4 -- State what is NOT already covered -- the actual marginal
thing that would need to be built.
Respond with ONLY the JSON object, no markdown fences."""
    return extract_json(gemini_generate(prompt))


def entailment_check(gap, cap, elaboration):
    prompt = f"""Gap's stated required operations: {gap.get('required_operations')}
Capability's stated core operations: {cap.get('core_operations')}

Stage 3 (overlap claimed): {elaboration.get('stage_3')}
Stage 4 (marginal build claimed): {elaboration.get('stage_4')}

Does everything in Stage 3 and Stage 4 trace back to the operations
listed above, or does it introduce something from neither? Respond
with ONLY a JSON object: {{"leakage_flagged": true or false, "notes":
"<one sentence>"}}"""
    return extract_json(gemini_generate(prompt))


def existence_check(elaboration):
    prompt = f"""Search: does a real, existing product or tool already
do what is described here?
{elaboration.get('stage_4')}

Respond with ONLY a JSON object: {{"already_exists": true or false,
"evidence": "<one or two sentences>"}}"""
    return extract_json(gemini_generate(prompt, use_search=True))


def perturbation_check(gap, cap, precedents):
    gap2 = copy.deepcopy(gap)
    sp = gap2["structural_properties"]
    original = sp.get("mutability")
    sp["mutability"] = f"OPPOSITE of stated: {original}"
    new_elab = elaborate(gap2, cap, precedents)
    return {"perturbed_field": "mutability", "new_stage_4": new_elab.get("stage_4")}


def adversarial_audit(elaboration):
    prompt = f"""You have zero investment in the following idea. Raise
exactly 3 "isn't it true that..." objections targeting hidden
assumptions or likely prior failure modes, answer each yourself, and
return a verdict.

IDEA: {elaboration.get('stage_4')}

Respond with ONLY a JSON object: {{"objections": [{{"objection": "...",
"answer": "..."}}, ...], "verdict": "survives" | "needs_revision" |
"fails"}}"""
    return extract_json(gemini_generate(prompt))


def main():
    print("Loading gaps, capabilities, and existing verdicts...", flush=True)
    gaps = load_jsonl(GAPS_PATH)
    caps = load_jsonl(CAPS_PATH)
    verdicts = load_jsonl(VERDICTS_PATH)
    seen_pairs = {(v["gap_id"], v["capability_id"]) for v in verdicts}
    print(f"Loaded {len(gaps)} gap(s), {len(caps)} capability(ies), {len(verdicts)} past verdict(s).", flush=True)

    print("Building shortlist by embedding similarity...", flush=True)
    candidates = find_candidates(gaps, caps, seen_pairs)
    if not candidates:
        print("No new candidate pairs. Done.", flush=True)
        return
    print(f"Shortlisted {len(candidates)} candidate pair(s) to judge.", flush=True)

    for i, (gap, cap, sim) in enumerate(candidates, 1):
        print(f"\n--- Candidate {i}/{len(candidates)}: {gap['id']} x {cap['id']} (similarity={sim:.3f}) ---", flush=True)

        print("  [1/6] Blind verdict scoring...", flush=True)
        verdict_result = blind_verdict(gap, cap)
        score = verdict_result.get("fit_score", 0)
        print(f"  [1/6] Fit score: {score} -- {verdict_result.get('justification', '')}", flush=True)

        record = {
            "id": new_id("pv"),
            "gap_id": gap["id"],
            "capability_id": cap["id"],
            "created_at": now_iso(),
            "retrieval_similarity": sim,
            "fit_score": score,
            "justification": verdict_result.get("justification"),
        }

        if score < THRESHOLD:
            record["status"] = "no_match"
            append_jsonl(VERDICTS_PATH, record)
            print(f"  Below threshold ({THRESHOLD}) -- writing no_match, moving on.", flush=True)
            continue

        print("  [2/6] Retrieving precedent from past verdicts...", flush=True)
        precedents = find_precedents(gap, cap, verdicts)
        print(f"  [2/6] Found {len(precedents)} relevant precedent(s).", flush=True)

        print("  [3/6] Running four-stage elaboration...", flush=True)
        elaboration = elaborate(gap, cap, precedents)
        record["elaboration"] = elaboration
        print(f"  [3/6] Marginal build proposed: {elaboration.get('stage_4', '')[:100]}", flush=True)

        print("  [4/6] Running entailment check...", flush=True)
        record["entailment_check"] = entailment_check(gap, cap, elaboration)
        print(f"  [4/6] Leakage flagged: {record['entailment_check'].get('leakage_flagged')}", flush=True)

        print("  [5/6] Running existence check (grounded search)...", flush=True)
        existence = existence_check(elaboration)
        record["existence_check"] = existence

        if existence.get("already_exists"):
            record["status"] = "existing_product_found"
            append_jsonl(VERDICTS_PATH, record)
            print("  [5/6] Already exists -- stopping here, writing existing_product_found.", flush=True)
            continue
        print("  [5/6] No existing product found, proceeding to audit.", flush=True)

        print("  [6/6] Running perturbation check and adversarial cross-exam...", flush=True)
        record["perturbation_check"] = perturbation_check(gap, cap, precedents)

        audit = adversarial_audit(elaboration)
        record["adversarial_audit"] = audit
        record["audit_verdict"] = audit.get("verdict")
        record["status"] = "audited"
        print(f"  [6/6] Audit verdict: {record['audit_verdict']}", flush=True)

        try:
            record["pair_embedding"] = gemini_embed(
                gap_public_fields(gap) + "\n" + cap_public_fields(cap)
            )
        except Exception:
            pass

        append_jsonl(VERDICTS_PATH, record)
        print(f"  Done. {record['id']}: score={score} -> audited, verdict={record['audit_verdict']}", flush=True)

    print("\nAll candidates processed.", flush=True)


if __name__ == "__main__":
    main()
