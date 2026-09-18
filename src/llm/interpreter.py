"""LLM-based viniyoga interpretation with RAG over the Abhinaya Darpana KB.

Given a detected mudra (or pair) and optional context, this:
1. Retrieves the top-k most relevant viniyoga entries from the KB
2. Constructs a structured prompt giving the LLM the canonical options
3. Streams a grounded interpretation back

Key design choices:
- LLM is constrained to interpret from retrieved viniyogas only — no
  free generation. This is the RAG part: the LLM does contextual selection
  and natural-language synthesis, not knowledge invention.
- System prompt explicitly tells the model to acknowledge uncertainty rather
  than confabulate when context is ambiguous. Hallucination eval (C9) will
  check this is actually happening.

Provider: Groq's free inference API serving Llama 3.3 70B. OpenAI-compatible
chat-completions API. Easy to swap to any other OpenAI-compatible provider
(OpenAI, Together, Fireworks, etc.) by changing the client init and model
name only.
"""
import os
from typing import AsyncIterator, Optional

from groq import AsyncGroq
from dotenv import load_dotenv

from src.kb.retrieval import ViniyogaRetriever

load_dotenv()

# NOTE: llama-3.3-70b-versatile was retired from Groq's free tier after this
# project's initial build. openai/gpt-oss-120b is the current closest
# equivalent (large open-weight MoE model, free tier, OpenAI-compatible API).
# Swappable by changing this constant only — see module docstring.
LLM_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a Bharatanatyam interpreter helping a dancer understand the viniyoga (traditional meanings) of detected mudras.

You will receive:
- One or more detected mudras (hand gestures)
- Optional context about hand position, body posture, or sequence
- A retrieved list of canonical viniyogas from Abhinaya Darpana

Your job:
- Interpret what the dancer is most likely expressing, grounded in the retrieved viniyogas
- Be concise: 2-3 sentences maximum
- Reference specific viniyogas by name when relevant
- If multiple viniyogas are equally plausible, say so — do not pick arbitrarily
- Never invent viniyogas not in the retrieved list

Tone: warm and informative, like a knowledgeable teacher. Avoid academic jargon.
"""

# Used only by the no-RAG arm of the faithfulness eval (scripts/eval_llm.py)
# to measure what the model does with no retrieved grounding at all — i.e.
# to quantify the value RAG actually adds, not to ship in the app.
NO_RAG_SYSTEM_PROMPT = """You are a Bharatanatyam interpreter helping a dancer understand the viniyoga (traditional meaning) of detected mudras.

You will receive one or more detected mudras and optional context about hand position or body posture. Using your own knowledge of Bharatanatyam and the Abhinaya Darpana, interpret what the dancer is most likely expressing.

Be concise: 2-3 sentences maximum. Tone: warm and informative, like a knowledgeable teacher.
"""


class ViniyogaInterpreter:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
            )
        self.client = AsyncGroq(api_key=api_key)
        self.retriever = ViniyogaRetriever()

    def build_user_message(
        self,
        mudras: list[str],
        context: Optional[str],
        retrieved: list[dict],
    ) -> str:
        retrieved_block = "\n".join(
            f"- {hit['metadata']['mudra']}: {hit['metadata']['meaning']} "
            f"({hit['metadata']['context']})"
            for hit in retrieved
        )

        ctx_line = f"\nContext: {context}" if context else ""
        mudras_line = ", ".join(mudras)

        return f"""Detected mudra(s): {mudras_line}{ctx_line}

Retrieved canonical viniyogas:
{retrieved_block}

What is the dancer most likely expressing?"""

    async def interpret_stream(
        self,
        mudras: list[str],
        context: Optional[str] = None,
        k: int = 6,
    ) -> AsyncIterator[str]:
        """Yield text chunks as the LLM generates them."""
        retrieved = self.retriever.retrieve(mudras, context=context, k=k)
        user_msg = self.build_user_message(mudras, context, retrieved)

        stream = await self.client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=300,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            stream=True,
        )
        async for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text

    async def interpret_stream_no_rag(
        self,
        mudras: list[str],
        context: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Same task, no retrieval — the model answers from parametric
        knowledge alone. Exists only to measure the RAG grounding effect
        in scripts/eval_llm.py; the app always uses interpret_stream."""
        ctx_line = f"\nContext: {context}" if context else ""
        mudras_line = ", ".join(mudras)
        user_msg = f"""Detected mudra(s): {mudras_line}{ctx_line}

What is the dancer most likely expressing?"""

        stream = await self.client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=300,
            messages=[
                {"role": "system", "content": NO_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            stream=True,
        )
        async for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text


async def _demo():
    """CLI demo: interpret a fixed example."""
    interp = ViniyogaInterpreter()
    print("Mudra: Pathaka | Context: arm raised above head\n")
    async for chunk in interp.interpret_stream(
        ["Pathaka"],
        context="arm raised above head",
    ):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_demo())
