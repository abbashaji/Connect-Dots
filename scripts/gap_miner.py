"""
Gap Miner. Runs standalone, with no access to capabilities.jsonl -- that's
not a permissions restriction, it's a deliberate isolation choice: this
script must never be *able* to pick a gap because it happens to pair well
with a capability, because it has no way of knowing what capabilities exist.
"""
from common import load_jsonl, append_jsonl, gemini_generate, gemini_embed, extract_json, new_id, now_iso

DATA_PATH = "data/gaps.jsonl"

SYSTEM = """You are the GAP MINER for an independent research pipeline.
Your only job is to find ONE unresolved technical limitation from a
recent, credible source: a paper's "Limitations" section, an unresolved
GitHub issue, a complaint thread, or a patent's "problem to be solved"
paragraph. Prefer sources you can confirm are recent via search. Pick
based on how real and unpriced the complaint looks -- repeated
complaints, described manual workarounds, no existing tooling -- not
because of any capability you might imagine pairing it with. You have
no knowledge of any capability corpus and must not speculate about one.
"""


def build_prompt(existing_complaints):
    avoid = "\n".join(f"- {c}" for c in existing_complaints[-30:]) or "(none yet)"
    return f"""Search for ONE new unresolved technical limitation not
already covered by these existing entries (avoid near-duplicates):
{avoid}

Respond with ONLY a JSON object (no markdown fences, no commentary),
matching exactly this shape:
{{
  "source": "url or citation",
  "source_date": "YYYY-MM or null",
  "artifact_type": "abstract data/process shape, no domain nouns",
  "structural_properties": {{
    "cardinality_scale": "order of magnitude, fixed or variable",
    "dimensionality": "R^d for what d, or unspecified",
    "production_method": "optimization / sampling / enumeration / manual authoring / sensor capture / ...",
    "mutability": "is it typically edited post-production, or consumed as-is"
  }},
  "complaint": "the specific downstream pain, stripped of domain vocabulary",
  "workaround": "what people do today to cope",
  "required_operations": ["op1", "op2", "op3"],
  "domain_tag": "the real field/product this is drawn from, private audit use only"
}}"""


def public_text(gap):
    sp = gap["structural_properties"]
    return (
        f"Artifact type: {gap['artifact_type']}\n"
        f"Cardinality/scale: {sp.get('cardinality_scale', '')}\n"
        f"Dimensionality: {sp.get('dimensionality', '')}\n"
        f"Production method: {sp.get('production_method', '')}\n"
        f"Mutability: {sp.get('mutability', '')}\n"
        f"Complaint: {gap['complaint']}\n"
        f"Workaround: {gap.get('workaround', '')}"
    )


def main():
    print("Loading existing gaps...", flush=True)
    existing = load_jsonl(DATA_PATH)
    existing_complaints = [g.get("complaint", "") for g in existing]
    print(f"Loaded {len(existing)} existing gap(s).", flush=True)

    print("Calling Gemini (grounded search) for a new gap candidate...", flush=True)
    raw = gemini_generate(build_prompt(existing_complaints), system=SYSTEM, use_search=True)
    print("Got a response, parsing JSON...", flush=True)
    data = extract_json(raw)

    gap = {"id": new_id("gap"), "created_at": now_iso(), **data}
    print(f"Parsed gap {gap['id']}: {gap.get('complaint', '')[:80]}", flush=True)

    print("Embedding gap for similarity search...", flush=True)
    gap["embedding"] = gemini_embed(public_text(gap))

    print(f"Writing {gap['id']} to {DATA_PATH}...", flush=True)
    append_jsonl(DATA_PATH, gap)
    print(f"Done. Wrote {gap['id']}: {gap['complaint'][:80]}", flush=True)


if __name__ == "__main__":
    main()
