"""
Shared helpers for the ideation pipeline (gap_miner.py, capability_miner.py,
judge.py). No state lives here beyond the Gemini API key read at import time
-- everything else is passed explicitly so each script stays a plain,
inspectable function call, not a hidden shared object.

Model choice and rate limits are pinned to this project's actual free-tier
quota (see My-free_Gemini_Api.txt): gemini-3.1-flash-lite (15 RPM) for
generation, gemini-embedding-001 (100 RPM) for embeddings. If you upgrade
tier or change models, update RATE_LIMIT_SECONDS_* to match the new RPM
(60 / RPM, with a small buffer).
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
GENERATE_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

# 15 RPM on the generate model -> one call every 4s minimum. Buffered to 4.5s.
RATE_LIMIT_SECONDS_GENERATE = 4.5
# 100 RPM on the embedding model -> one call every 0.6s minimum. Buffered to 1s.
RATE_LIMIT_SECONDS_EMBED = 1.0

_last_generate_call = 0.0
_last_embed_call = 0.0


def _throttle(kind):
    global _last_generate_call, _last_embed_call
    now = time.monotonic()
    if kind == "generate":
        wait = RATE_LIMIT_SECONDS_GENERATE - (now - _last_generate_call)
        if wait > 0:
            time.sleep(wait)
        _last_generate_call = time.monotonic()
    else:
        wait = RATE_LIMIT_SECONDS_EMBED - (now - _last_embed_call)
        if wait > 0:
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
    _throttle("embed")
    url = f"{GEMINI_BASE}/models/{EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = {
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def gemini_generate(prompt, system=None, use_search=False, retries=3):
    """Returns raw text. Callers that expect JSON should pass it through
    extract_json() themselves -- kept separate so a caller can inspect the
    raw text on parse failure instead of losing it inside this function.

    No temperature/top_p/top_k override: Gemini 3.x's own docs recommend
    leaving these at default, since the model's reasoning is tuned for
    them -- setting a custom value here would work against the model,
    not with it.
    """
    url = f"{GEMINI_BASE}/models/{GENERATE_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if use_search:
        payload["tools"] = [{"googleSearch": {}}]

    last_err = None
    for attempt in range(retries):
        _throttle("generate")
        try:
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code == 429:
                # Quota hit despite throttling (e.g. another workflow run
                # used up the window) -- back off hard, not just the
                # standard retry delay.
                time.sleep(30 * (attempt + 1))
                resp.raise_for_status()
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:  # noqa: BLE001 - deliberately broad, we retry regardless of cause
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gemini_generate failed after {retries} attempts: {last_err}")
