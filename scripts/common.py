"""
Shared helpers for the ideation pipeline (gap_miner.py, capability_miner.py,
judge.py). No state lives here beyond the Gemini API key read at import time
-- everything else is passed explicitly so each script stays a plain,
inspectable function call, not a hidden shared object.

Two generate models are used, split by whether a call needs grounded
search:
  - GENERATE_MODEL (gemini-3.5-flash-lite, 15 RPM / 500 RPD): the only
    one used for calls with use_search=True. Grounding is not supported
    on Gemma, so use_search forces this model regardless of what's asked.
  - GEMMA_MODEL (gemma-4-31b-it, 30 RPM / 14,400 RPD per current quota
    sheet): used for everything that doesn't need search, to keep those
    calls off the much scarcer 500 RPD budget.
Embeddings use gemini-embedding-001 (100 RPM / 1,000 RPD), untouched by
this split since it was never the bottleneck.
"""
import os
import re
import json
import math
import time
import uuid
import datetime
import requests

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GENERATE_MODEL = "gemini-3.5-flash-lite"
GEMMA_MODEL = "gemma-4-31b-it"
EMBED_MODEL = "gemini-embedding-001"

# Per-model minimum seconds between calls, derived from 60/RPM with a
# small buffer. Update these if the quota sheet changes.
RATE_LIMIT_SECONDS = {
    GENERATE_MODEL: 4.5,   # 15 RPM
    GEMMA_MODEL: 2.2,      # 30 RPM
}
RATE_LIMIT_SECONDS_EMBED = 1.0  # 100 RPM

_last_call_by_model = {}
_last_embed_call = 0.0


def _throttle_model(model):
    now = time.monotonic()
    last = _last_call_by_model.get(model, 0.0)
    limit = RATE_LIMIT_SECONDS.get(model, 4.5)  # safe default if an unlisted model is passed
    wait = limit - (now - last)
    if wait > 0:
        print(f"    [throttle:{model}] waiting {wait:.1f}s to respect rate limit...", flush=True)
        time.sleep(wait)
    _last_call_by_model[model] = time.monotonic()


def _throttle_embed():
    global _last_embed_call
    now = time.monotonic()
    wait = RATE_LIMIT_SECONDS_EMBED - (now - _last_embed_call)
    if wait > 0:
        print(f"    [throttle:embed] waiting {wait:.1f}s to respect rate limit...", flush=True)
        time.sleep(wait)
    _last_embed_call = time.monotonic()


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def extract_json(text):
    """Gemini sometimes wraps JSON in ```json fences or adds stray prose.
    Strip fences first, then fall back to grabbing the first {...} block."""
    if isinstance(text, (dict, list)):
        return text
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def gemini_embed(text, task_type="RETRIEVAL_DOCUMENT"):
    print(f"    [embed] text ({len(text)} chars): {text[:150]!r}...", flush=True)
    _throttle_embed()
    url = f"{GEMINI_BASE}/models/{EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = {
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
    }
    print(f"    [embed] POST {url.split('?')[0]}", flush=True)
    t0 = time.monotonic()
    resp = requests.post(url, json=payload, timeout=60)
    elapsed = time.monotonic() - t0
    print(f"    [embed] status={resp.status_code} elapsed={elapsed:.1f}s", flush=True)
    resp.raise_for_status()
    values = resp.json()["embedding"]["values"]
    print(f"    [embed] got vector of length {len(values)}", flush=True)
    return values


def gemini_generate(prompt, system=None, use_search=False, model=None, retries=3):
    """Returns raw text. Callers that expect JSON should pass it through
    extract_json() themselves -- kept separate so a caller can inspect the
    raw text on parse failure instead of losing it inside this function.

    model: defaults to GENERATE_MODEL. Pass GEMMA_MODEL explicitly for
    calls that don't need search, to keep them off the scarcer RPD
    budget. If use_search=True, GENERATE_MODEL is forced regardless of
    what's passed here -- grounding isn't supported on Gemma.

    No temperature/top_p/top_k override: Gemini 3.x's own docs recommend
    leaving these at default, since the model's reasoning is tuned for
    them -- setting a custom value here would work against the model,
    not with it. (Gemma has no such documented guidance either way, so
    the same no-override approach is applied uniformly.)
    """
    resolved_model = model or GENERATE_MODEL
    if use_search and resolved_model != GENERATE_MODEL:
        print(f"    [generate] search requested with model={resolved_model} -- "
              f"forcing {GENERATE_MODEL} since grounding needs a real Gemini model.", flush=True)
        resolved_model = GENERATE_MODEL

    url = f"{GEMINI_BASE}/models/{resolved_model}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if use_search:
        payload["tools"] = [{"googleSearch": {}}]

    print(f"    [generate] model={resolved_model} search={use_search} prompt ({len(prompt)} chars):", flush=True)
    print(f"    [generate] >>> {prompt[:300]!r}...", flush=True)

    last_err = None
    for attempt in range(retries):
        _throttle_model(resolved_model)
        resp = None
        try:
            print(f"    [generate] POST attempt {attempt + 1}/{retries} -> {url.split('?')[0]}", flush=True)
            t0 = time.monotonic()
            resp = requests.post(url, json=payload, timeout=120)
            elapsed = time.monotonic() - t0
            print(f"    [generate] status={resp.status_code} elapsed={elapsed:.1f}s", flush=True)

            if resp.status_code == 429:
                print(f"    [generate] 429 body: {resp.text[:800]!r}", flush=True)
                backoff = 30 * (attempt + 1)
                print(f"    [generate] backing off {backoff}s...", flush=True)
                time.sleep(backoff)
                resp.raise_for_status()
            resp.raise_for_status()

            data = resp.json()
            candidate = data["candidates"][0]
            finish_reason = candidate.get("finishReason", "?")
            grounding = candidate.get("groundingMetadata")
            print(f"    [generate] finishReason={finish_reason} grounded={'yes' if grounding else 'no'}", flush=True)
            if grounding and grounding.get("webSearchQueries"):
                print(f"    [generate] search queries used: {grounding['webSearchQueries']}", flush=True)

            text = candidate["content"]["parts"][0]["text"]
            print(f"    [generate] <<< raw response ({len(text)} chars): {text[:300]!r}...", flush=True)
            return text
        except Exception as e:  # noqa: BLE001 - deliberately broad, we retry regardless of cause
            last_err = e
            print(f"    [generate] attempt {attempt + 1} failed: {e!r}", flush=True)
            if resp is not None and attempt == retries - 1:
                print(f"    [generate] final raw body: {resp.text[:500]!r}", flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gemini_generate failed after {retries} attempts: {last_err}")
