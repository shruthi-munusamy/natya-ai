"""LLM-based viniyoga interpretation with RAG over the Abhinaya Darpana KB.

Given a detected mudra (or pair) and optional context, this:
1. Retrieves the top-k most relevant viniyoga entries from the KB
2. Constructs a structured prompt giving the LLM the canonical options
3. Streams a grounded interpretation back

The model is told to interpret only from the retrieved viniyogas and to
say so when several are equally plausible, rather than invent meanings.
scripts/eval_llm.py measures how well it sticks to that.

Provider: Groq's OpenAI-compatible chat-completions API. Swapping providers
means changing the client init and LLM_MODEL.
"""
import os
from typing import AsyncIterator, Optional

from groq import AsyncGroq
from dotenv import load_dotenv

from src.kb.retrieval import ViniyogaRetriever

load_dotenv()

# Replaced llama-3.3-70b-versatile after Groq retired it from the free tier.
LLM_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a Bharatanatyam interpreter helping a dancer understand the viniyoga (traditional meanings) of detected mudras.

You will receive:
- One or more detected mudras (hand gestures)
- Optional context about hand position, body posture, or sequence
- A retrieved list of canonical viniyogas from Abhinaya Darpana

Your job:
- Interpret what the dancer is most likely expressing, grounded in the retrieved viniyogas
- Be concise: 2-3 sentences maximum, written as plain prose with no lists, bullets or markdown
- Reference specific viniyogas by name when relevant
- If multiple viniyogas are equally plausible, say so — do not pick arbitrarily
- Never invent viniyogas not in the retrieved list
- The retrieved contexts describe how each meaning is traditionally performed, not what this dancer is doing. Only mention where the dancer's hand or body is if it is given under Context; otherwise don't describe their position

Tone: warm and informative, like a knowledgeable teacher. Avoid academic jargon.
"""

# Baseline for scripts/eval_llm.py: the same task with no retrieved grounding.
NO_RAG_SYSTEM_PROMPT = """You are a Bharatanatyam interpreter helping a dancer understand the viniyoga (traditional meaning) of detected mudras.

You will receive one or more detected mudras and optional context about hand position or body posture. Using your own knowledge of Bharatanatyam and the Abhinaya Darpana, interpret what the dancer is most likely expressing.

Be concise: 2-3 sentences maximum, written as plain prose with no lists, bullets or markdown. Tone: warm and informative, like a knowledgeable teacher.
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
        async for text in self._stream(SYSTEM_PROMPT, user_msg):
            yield text

    async def interpret_stream_no_rag(
        self,
        mudras: list[str],
        context: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Same task without retrieval. Used only by scripts/eval_llm.py."""
        ctx_line = f"\nContext: {context}" if context else ""
        mudras_line = ", ".join(mudras)
        user_msg = f"""Detected mudra(s): {mudras_line}{ctx_line}

What is the dancer most likely expressing?"""
        async for text in self._stream(NO_RAG_SYSTEM_PROMPT, user_msg):
            yield text

    async def _stream(self, system_prompt: str, user_msg: str) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=LLM_MODEL,
            # gpt-oss reasons before answering, and those tokens count
            # against max_tokens; at 300 longer answers were cut off.
            max_tokens=1024,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            stream=True,
        )
        async for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text


async def _demo():
    interp = ViniyogaInterpreter()
    print("Mudra: Pataka | Context: arm raised above head\n")
    async for chunk in interp.interpret_stream(
        ["Pataka"],
        context="arm raised above head",
    ):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_demo())
