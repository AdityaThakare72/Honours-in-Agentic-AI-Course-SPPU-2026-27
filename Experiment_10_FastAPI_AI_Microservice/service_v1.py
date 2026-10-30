import os, time, json, asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))
app = FastAPI(title="LLM microservice", version="1")
llm = AsyncOpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta/openai/", max_retries=2)
MODEL = "gemini-2.5-flash-lite"


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    max_tokens: int = Field(200, ge=1, le=1000)
    temperature: float = Field(0.2, ge=0, le=1)


class ChatResponse(BaseModel):
    answer: str
    model: str
    latency_ms: float
    total_tokens: int


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t0 = time.time()
    try:
        r = await llm.chat.completions.create(model=MODEL, max_tokens=req.max_tokens, temperature=req.temperature,
                                              messages=[{"role": "user", "content": req.prompt}])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream model error: {type(e).__name__}")
    return ChatResponse(answer=r.choices[0].message.content or "", model=MODEL,
                        latency_ms=round((time.time() - t0) * 1000, 1), total_tokens=r.usage.total_tokens)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def events():
        try:
            stream = await llm.chat.completions.create(model=MODEL, max_tokens=req.max_tokens, temperature=req.temperature, stream=True,
                                                       messages=[{"role": "user", "content": req.prompt}])
            async for chunk in stream:
                piece = chunk.choices[0].delta.content if chunk.choices else ""
                if piece:
                    yield f"data: {json.dumps({'token': piece})}\n\n"
        except Exception as e:                                   # headers are already sent, so report the error inside the stream
            yield f"data: {json.dumps({'error': type(e).__name__})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/work-async")
async def work_async(seconds: float = 0.5):
    await asyncio.sleep(seconds)                 # cooperative wait: other requests run meanwhile
    return {"ok": True}


@app.get("/work-sync")
def work_sync(seconds: float = 0.5):
    time.sleep(seconds)                          # plain def: FastAPI runs it in a worker thread
    return {"ok": True}


@app.get("/work-blocking")
async def work_blocking(seconds: float = 0.5):
    time.sleep(seconds)                          # BUG: blocks the event loop inside async def
    return {"ok": True}
