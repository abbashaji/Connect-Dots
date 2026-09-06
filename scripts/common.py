"""
Shared helpers for the ideation pipeline (gap_miner.py, capability_miner.py,
judge.py). No state lives here beyond the Gemini API key read at import time
-- everything else is passed explicitly so each script stays a plain,
inspectable function call, not a hidden shared object.
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
GENERATE_MODEL = "gemini-2.0-flash"
EMBED_MODEL = "gemini-embedding-001"


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
    url = f"{GEMINI_BASE}/models/{EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = {
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def gemini_generate(prompt, system=None, use_search=False, temperature=0.6, retries=3):
    """Returns raw text. Callers that expect JSON should pass it through
    extract_json() themselves -- kept separate so a caller can inspect the
    raw text on parse failure instead of losing it inside this function."""
    url = f"{GEMINI_BASE}/models/{GENERATE_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if use_search:
        payload["tools"] = [{"googleSearch": {}}]

    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:  # noqa: BLE001 - deliberately broad, we retry regardless of cause
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gemini_generate failed after {retries} attempts: {last_err}")
