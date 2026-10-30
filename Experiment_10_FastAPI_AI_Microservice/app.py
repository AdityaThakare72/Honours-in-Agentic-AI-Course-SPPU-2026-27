import os, json, time, uuid, asyncio
from collections import deque
from contextlib import asynccontextmanager, contextmanager
import numpy as np
import redis
from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
from semcache import SemanticCache

load_dotenv(find_dotenv(usecwd=True))
MODEL = "gemini-2.5-flash-lite"
API_KEY = os.environ.get("SERVICE_API_KEY", "dev-key")
THRESHOLD = float(os.environ.get("CACHE_THRESHOLD", "0.8"))
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "120"))
MOCK = os.environ.get("MOCK_LLM") == "1"                      # canned answers: lets you run the service without an API key
SYSTEM = "Answer in at most three short sentences."

if os.environ.get("LANGSMITH_TRACING") == "true":             # send traces to LangSmith when an account is configured
    from langsmith import traceable
else:
    def traceable(**kw):
        return lambda f: f


def log_event(**kw):
    with open("requests.jsonl", "a") as f:
        f.write(json.dumps({"ts": round(time.time(), 3), **kw}) + "\n")


class Tracer:
    # records a tree of timed spans for one request in traces.jsonl
    def __init__(self, trace_id):
        self.id, self.root, self.stack, self.t0 = trace_id, uuid.uuid4().hex[:8], [], time.time()

    def _write(self, sid, parent, name, t, attrs):
        with open("traces.jsonl", "a") as f:
            f.write(json.dumps({"trace_id": self.id, "span_id": sid, "parent_id": parent, "name": name,
                                "start_ms": round((t - self.t0) * 1000, 2), "ms": round((time.time() - t) * 1000, 2), "attrs": attrs}) + "\n")

    @contextmanager
    def span(self, name, **attrs):
        sid, parent, t = uuid.uuid4().hex[:8], (self.stack[-1] if self.stack else self.root), time.time()
        self.stack.append(sid)
        try:
            yield attrs
        finally:
            self.stack.pop()
            self._write(sid, parent, name, t, attrs)

    def finish(self, **attrs):
        self._write(self.root, None, "request", self.t0, attrs)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    max_tokens: int = Field(150, ge=1, le=600)
    temperature: float = Field(0.0, ge=0, le=1)
    use_cache: bool = True


@asynccontextmanager
async def lifespan(app):
    app.state.r = redis.Redis(host="127.0.0.1", port=6379)
    app.state.cache = SemanticCache(app.state.r, SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
                                    threshold=THRESHOLD, ttl=3600, namespace="svc:cache")
    app.state.llm = None if MOCK else AsyncOpenAI(api_key=os.environ["GEMINI_API_KEY"],
                                                  base_url="https://generativelanguage.googleapis.com/v1beta/openai/", max_retries=2)
    app.state.lat = {"hit": deque(maxlen=2000), "miss": deque(maxlen=2000), "bypass": deque(maxlen=2000)}
    app.state.errors = 0
    yield


app = FastAPI(title="LLM service with semantic cache", lifespan=lifespan)


@app.middleware("http")
async def request_id_and_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id, t0 = rid, time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        log_event(kind="http", request_id=rid, path=request.url.path, status=500, error=type(e).__name__)
        return JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500, headers={"X-Request-ID": rid})
    response.headers["X-Request-ID"] = rid
    log_event(kind="http", request_id=rid, method=request.method, path=request.url.path, status=response.status_code,
              ttfb_ms=round((time.time() - t0) * 1000, 1))
    return response


async def guard(request: Request, x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    key = f"svc:rl:{x_api_key}:{int(time.time() // 60)}"
    n = await run_in_threadpool(lambda: (request.app.state.r.incr(key), request.app.state.r.expire(key, 90))[0])
    if n > RATE_LIMIT:
        raise HTTPException(status_code=429, detail="rate limit exceeded", headers={"Retry-After": "30"})


@traceable(name="gemini_stream", run_type="llm")
async def llm_stream(llm, prompt, max_tokens, temperature):
    if MOCK:
        for w in f"[mock answer] You asked: {prompt[:80]}".split():
            await asyncio.sleep(0.05)
            yield w + " "
        return
    stream = await llm.chat.completions.create(model=MODEL, max_tokens=max_tokens, temperature=temperature, stream=True,
                                               messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    async for chunk in stream:
        piece = chunk.choices[0].delta.content if chunk.choices else ""
        if piece:
            yield piece


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, _: None = Depends(guard)):
    rid, st = request.state.request_id, request.app.state
    tr, t0, answer, score, vec, status = Tracer(rid), time.time(), None, 0.0, None, "bypass"
    if req.use_cache:
        with tr.span("cache.lookup") as a:
            with tr.span("embed"):
                vec = await run_in_threadpool(st.cache.embed, req.prompt)
            answer, score, _ = await run_in_threadpool(st.cache.lookup, req.prompt, vec)
            a["hit"], a["score"] = answer is not None, round(score, 3)
        status = "hit" if answer is not None else "miss"

    async def events():
        full, first, err = [], None, None
        try:
            yield sse({"meta": {"request_id": rid, "cache": status, "score": round(score, 3)}})
            if answer is not None:
                words = answer.split(" ")
                for i in range(0, len(words), 4):                     # replay the stored answer in small pieces
                    first = first or time.time() - t0
                    yield sse({"token": " ".join(words[i:i + 4]) + " "})
            else:
                with tr.span("llm.stream", model="mock" if MOCK else MODEL) as a:
                    async for piece in llm_stream(st.llm, req.prompt, req.max_tokens, req.temperature):
                        first = first or time.time() - t0
                        full.append(piece)
                        yield sse({"token": piece})
                    a["chars"] = len("".join(full))
                if req.use_cache and full:
                    with tr.span("cache.store"):
                        await run_in_threadpool(st.cache.store, req.prompt, "".join(full).strip(), vec)
            yield "data: [DONE]\n\n"
        except Exception as e:
            err = type(e).__name__
            st.errors += 1
            yield sse({"error": err})
        finally:
            total = (time.time() - t0) * 1000
            st.lat[status].append(total)
            tr.finish(cache=status, error=err)
            log_event(kind="chat", request_id=rid, cache=status, score=round(score, 3), ttft_ms=round((first or 0) * 1000, 1),
                      total_ms=round(total, 1), error=err, prompt_chars=len(req.prompt))

    return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Cache": status, "X-Cache-Score": f"{score:.3f}"})


@app.get("/health")
async def health():
    return {"status": "ok", "model": "mock" if MOCK else MODEL, "threshold": THRESHOLD}


@app.get("/metrics")
async def metrics(request: Request, _: None = Depends(guard)):
    st = request.app.state
    pct = lambda d, p: round(float(np.percentile(d, p)), 1) if len(d) else None
    n = {k: len(v) for k, v in st.lat.items()}
    answered = n["hit"] + n["miss"]
    return {"requests": n, "errors": st.errors, "hit_rate": round(n["hit"] / answered, 3) if answered else 0.0,
            "p50_ms": {k: pct(v, 50) for k, v in st.lat.items()}, "p95_ms": {k: pct(v, 95) for k, v in st.lat.items()},
            "cache": st.cache.stats()}


@app.post("/cache/clear")
async def cache_clear(request: Request, _: None = Depends(guard)):
    await run_in_threadpool(request.app.state.cache.clear)
    return {"cleared": True}
