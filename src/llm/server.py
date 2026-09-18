"""FastAPI server exposing the viniyoga interpreter over SSE.

Why SSE instead of WebSockets: the client only needs a one-shot stream of
text tokens per interpretation request, not bidirectional messaging. SSE is
simpler to consume from the browser (native EventSource-style semantics via
fetch + ReadableStream) and works over plain HTTP, which keeps the free-tier
deploy story simple (no websocket upgrade handling needed on Render).
"""
import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.llm.interpreter import ViniyogaInterpreter

app = FastAPI(title="Natya viniyoga interpreter")

# Local dev origins plus the deployed Vercel origin. ALLOWED_ORIGINS can be
# set as a comma-separated env var on Render so the deployed frontend URL
# doesn't need a code change + redeploy to unblock CORS.
DEFAULT_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://natya-ai.vercel.app",
]
extra_origins = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = DEFAULT_ORIGINS + [o.strip() for o in extra_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

interpreter = ViniyogaInterpreter()


@app.on_event("startup")
def build_index_if_empty():
    # The chroma index is a derived artifact (rebuilt from data/kb/viniyoga.json
    # in a couple seconds), so it isn't committed to git. On a fresh deploy
    # with no persistent disk (Render free tier), this guarantees retrieval
    # works instead of silently returning zero results.
    if interpreter.retriever.collection.count() == 0:
        n = interpreter.retriever.build_index()
        print(f"Built viniyoga index from scratch: {n} entries")


class InterpretRequest(BaseModel):
    mudras: list[str]
    context: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/interpret/stream")
async def interpret_stream(req: InterpretRequest):
    async def event_generator():
        async for chunk in interpreter.interpret_stream(req.mudras, req.context):
            yield {"event": "token", "data": chunk}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())
