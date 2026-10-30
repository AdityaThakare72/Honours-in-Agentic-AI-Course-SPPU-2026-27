"""Shared helpers for the Experiment 7 notebooks: Gemini vision calls with a disk cache, and the three VQA metrics.

Every Gemini answer is stored in llm_cache.json, keyed by model, prompt, image bytes and resize setting.
Re-running a notebook then costs no API calls. Delete the file to query the API afresh.
"""
import os, io, re, json, time, base64, hashlib, string, threading
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv(find_dotenv(usecwd=True))
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CACHE_PATH = "llm_cache.json"
_cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
_client = None
_lock = threading.Lock()


def client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_URL, timeout=90, max_retries=0)
    return _client


def image_to_data_uri(path, max_side=None):
    """Read an image file and return a base64 data URI. max_side shrinks the longer side (for resolution experiments)."""
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        r = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(path, question, model="gemini-2.5-flash-lite", system=None, max_side=None, max_tokens=300):
    """Ask one question about one image. Returns dict(answer, prompt_tokens, completion_tokens, seconds, cached)."""
    uri = image_to_data_uri(path, max_side)
    key = hashlib.sha256(f"{model}|{system}|{question}|{max_tokens}|{uri}".encode()).hexdigest()
    if key in _cache:
        return {**_cache[key], "cached": True}
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": [{"type": "text", "text": question},
                                     {"type": "image_url", "image_url": {"url": uri}}]}]
    extra = {"reasoning_effort": "none"} if model == "gemini-2.5-flash" else {}
    for attempt in range(6):
        try:
            t = time.time()
            r = client().chat.completions.create(model=model, messages=messages, temperature=0,
                                                 max_tokens=max_tokens, **extra)
            out = dict(answer=(r.choices[0].message.content or "").strip(), prompt_tokens=r.usage.prompt_tokens,
                       completion_tokens=r.usage.completion_tokens, seconds=round(time.time() - t, 2))
            with _lock:
                _cache[key] = out
                json.dump(_cache, open(CACHE_PATH, "w"))
            return {**out, "cached": False}
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(4 * (attempt + 1))   # back off on rate limits and transient errors


# ---------------------------------------------------------------- metrics
def normalize(s):
    """Lowercase, drop punctuation and articles, collapse spaces."""
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation.replace(".", "")))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.strip(" .")
    return re.sub(r"\s+", " ", s).strip()


def vqa_accuracy(pred, answers):
    """VQAv2 soft accuracy: full credit if at least 3 of the 10 human annotators gave the same answer."""
    p = normalize(pred)
    matches = sum(normalize(a) == p for a in answers)
    return min(1.0, matches / 3)


def _to_number(s):
    s = str(s).replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def relaxed_accuracy(pred, gold, tol=0.05):
    """ChartQA relaxed accuracy: numbers may be off by 5 percent, text must match exactly after normalising."""
    a, b = _to_number(pred), _to_number(gold)
    if a is not None and b is not None:
        return float(a == b) if b == 0 else float(abs(a - b) / abs(b) <= tol)
    return float(normalize(pred) == normalize(gold))


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred, answers, tau=0.5):
    """DocVQA ANLS: 1 minus the normalised edit distance to the closest gold answer, zeroed when the distance is 0.5 or more."""
    best = 0.0
    p = normalize(pred)
    for g in answers:
        g = normalize(g)
        n = max(len(p), len(g)) or 1
        d = _lev(p, g) / n
        best = max(best, 1 - d if d < tau else 0.0)
    return best


def score(kind, pred, answers, gold):
    """Pick the metric that belongs to each dataset."""
    if kind == "photo":
        return vqa_accuracy(pred, answers)
    if kind == "chart":
        return relaxed_accuracy(pred, gold)
    return anls(pred, answers)
