"""
Capability Miner. Mirror image of gap_miner.py, and just as deliberately
isolated: this script must never read gaps.jsonl. It does not know a gap
corpus exists.
"""
from common import load_jsonl, append_jsonl, gemini_generate, gemini_embed, extract_json, new_id, now_iso

DATA_PATH = "data/capabilities.jsonl"

SYSTEM = """You are the CAPABILITY MINER for an independent research
pipeline. Your only job is to find ONE mature, general-purpose
platform, engine, or library and document its generic capability list.
Prefer boringly general-purpose, well-established infrastructure over
anything novel -- you want dormant, reusable capability, not cutting-
edge tech. Pick independent of any particular problem; you have no
knowledge of any gap corpus and must not speculate about one.
"""


def build_prompt(existing_capabilities):
    avoid = "\n".join(f"- {c}" for c in existing_capabilities[-30:]) or "(none yet)"
    return f"""Search for ONE mature, general-purpose platform/engine/
library not already covered by these existing entries (avoid
near-duplicates):
{avoid}

Respond with ONLY a JSON object (no markdown fences, no commentary),
matching exactly this shape:
{{
  "source": "url or citation",
  "operates_on": "abstract data shape it was built for",
  "core_operations": {{
    "selection": "modes it supports -- region-bound, freeform, single-pick, etc.",
    "transformation": "what can be done to a selected subset",
    "deletion_reduction": "yes/no, and how",
    "io": "what it can ingest/export, and in what generality",
    "delivery_environment": "where/how it runs"
  }},
  "cheap_to_extend_reason": "the hard 80% this capability already solved",
  "known_extensions": ["other domains it has already been repurposed for, if any"],
  "domain_tag": "the real product/org this is drawn from, private audit use only"
}}"""


def public_text(cap):
    co = cap["core_operations"]
    return (
        f"Operates on: {cap['operates_on']}\n"
        f"Selection: {co.get('selection', '')}\n"
        f"Transformation: {co.get('transformation', '')}\n"
        f"Deletion/reduction: {co.get('deletion_reduction', '')}\n"
        f"I/O: {co.get('io', '')}\n"
        f"Delivery environment: {co.get('delivery_environment', '')}\n"
        f"Why cheap to extend: {cap.get('cheap_to_extend_reason', '')}"
    )


def main():
    existing = load_jsonl(DATA_PATH)
    existing_summaries = [c.get("operates_on", "") for c in existing]

    raw = gemini_generate(build_prompt(existing_summaries), system=SYSTEM, use_search=True)
    data = extract_json(raw)

    cap = {"id": new_id("cap"), "created_at": now_iso(), **data}
    cap["embedding"] = gemini_embed(public_text(cap))

    append_jsonl(DATA_PATH, cap)
    print(f"Wrote {cap['id']}: {cap['operates_on'][:80]}")


if __name__ == "__main__":
    main()
