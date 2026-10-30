"""Shared building blocks for the Experiment 9 notebooks: three tools, a Gemini chat model and two disk caches.

- calculator: a small, safe arithmetic evaluator (no eval of arbitrary code).
- run_python: runs a snippet in a separate Python process with a time limit.
- web_search: DuckDuckGo text search through the ddgs package (no API key).

Search results are saved in search_cache.json and model responses in llm_cache.json, so re-running a notebook gives the same
answers without internet or an API key. Delete the two files to run everything live.
"""
import os, ast, json, math, time, hashlib, operator, subprocess, sys, tempfile, warnings
from dotenv import load_dotenv, find_dotenv
from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.load import dumpd, load
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv(usecwd=True))
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


# ------------------------------------------------------------------ caches
class JsonFileCache(BaseCache):
    """LangChain cache that keeps model responses in one JSON file."""

    def __init__(self, path="llm_cache.json"):
        self.path = path
        self.data = json.load(open(path)) if os.path.exists(path) else {}

    def _key(self, prompt, llm_string):
        return hashlib.sha256((llm_string + "||" + prompt).encode()).hexdigest()

    def lookup(self, prompt, llm_string):
        item = self.data.get(self._key(prompt, llm_string))
        return [load(g, allowed_objects="core") for g in item] if item else None

    def update(self, prompt, llm_string, return_val):
        self.data[self._key(prompt, llm_string)] = [dumpd(g) for g in return_val]
        json.dump(self.data, open(self.path, "w"))

    def clear(self, **kwargs):
        self.data = {}
        json.dump(self.data, open(self.path, "w"))


def enable_cache(path="llm_cache.json"):
    set_llm_cache(JsonFileCache(path))


def get_llm(model="gemini-2.5-flash-lite", temperature=0.0):
    """Gemini through its OpenAI-compatible endpoint, as a LangChain chat model."""
    return ChatOpenAI(model=model, temperature=temperature, api_key=os.environ["GEMINI_API_KEY"],
                      base_url=GEMINI_URL, timeout=60, max_retries=4)


# ------------------------------------------------------------------ calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "sin": math.sin, "cos": math.cos,
          "tan": math.tan, "factorial": math.factorial, "abs": abs, "round": round}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 1000:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError(f"not allowed: {ast.dump(node)[:40]}")


def calc(expression: str):
    """Evaluate a plain arithmetic expression. Only numbers, + - * / ** % //, a few math functions and pi/e are allowed."""
    return _eval(ast.parse(expression, mode="eval").body)


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, for example '48271 * 9371' or 'sqrt(2)' or '0.15 * 2480 + 7 ** 2'.
    Use it for any calculation with numbers. Supports + - * / ** % //, sqrt, log, log10, sin, cos, tan, factorial, pi, e."""
    try:
        return str(calc(expression))
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ------------------------------------------------------------------ code execution
def run_code(code: str, timeout: int = 5, max_chars: int = 1500) -> str:
    """Run a Python snippet in a separate process (isolated mode, temp directory, time limit). Not a real security sandbox."""
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return f"Error: the code did not finish within {timeout} seconds."
    out = (r.stdout + r.stderr).strip()
    return (out[:max_chars] + "...") if len(out) > max_chars else (out or "(no output; use print to see results)")


@tool
def run_python(code: str) -> str:
    """Run Python code and return what it prints. Use it for date arithmetic, string handling, loops, primes, data
    processing or anything that is easier to compute with a program. The code must print its result."""
    return run_code(code)


# ------------------------------------------------------------------ web search
SEARCH_CACHE = "search_cache.json"
_search_cache = json.load(open(SEARCH_CACHE)) if os.path.exists(SEARCH_CACHE) else {}


def search(query: str, k: int = 3):
    """DuckDuckGo text search. Results are cached by query."""
    if query not in _search_cache:
        from ddgs import DDGS
        for attempt in range(3):                    # the free search backend rate-limits now and then
            try:
                hits = DDGS().text(query, max_results=k)
                break
            except Exception as e:
                if attempt == 2:
                    return [dict(title="search failed", body=f"{type(e).__name__}", href="")]
                time.sleep(2 * (attempt + 1))
        _search_cache[query] = [dict(title=h["title"], body=h["body"], href=h["href"]) for h in hits]
        json.dump(_search_cache, open(SEARCH_CACHE, "w"))
    return _search_cache[query]


@tool
def web_search(query: str) -> str:
    """Search the web and return the top results (title, snippet, link). Use it for facts about people, companies,
    software, events or anything that may not be in your memory or may have changed recently."""
    hits = search(query)
    return "\n".join(f"- {h['title']}: {h['body'][:300]} ({h['href']})" for h in hits)


TOOLS = [calculator, run_python, web_search]
