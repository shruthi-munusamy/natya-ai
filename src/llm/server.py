"""FastAPI server exposing the viniyoga interpreter over SSE.

SSE rather than WebSockets: each request is a one-way token stream, and SSE
works over plain HTTP with no upgrade handling on Render.
"""
import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.llm.interpreter import ViniyogaInterpreter

app = FastAPI(title="Natya viniyoga interpreter")

# ALLOWED_ORIGINS (comma-separated) adds origins without a code change.
DEFAULT_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://natya-ai.vercel.app",
    "https://natya-ai-fawn.vercel.app",
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
    # The Chroma index isn't committed, and Render's free tier has no
    # persistent disk, so rebuild it from the KB JSON on a fresh deploy.
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
