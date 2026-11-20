import asyncio
from types import SimpleNamespace

import pytest

from src.llm import interpreter as interp_mod


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        async def chunks():
            for text in ["a", None, "b"]:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

        return chunks()


class _FakeRetriever:
    def retrieve(self, mudras, context=None, k=5):
        return [{"metadata": {"mudra": "Pataka", "meaning": "cloud", "context": "raised"}}]


@pytest.fixture
def interpreter():
    obj = interp_mod.ViniyogaInterpreter.__new__(interp_mod.ViniyogaInterpreter)
    obj.completions = _FakeCompletions()
    obj.client = SimpleNamespace(chat=SimpleNamespace(completions=obj.completions))
    obj.retriever = _FakeRetriever()
    return obj


async def _collect(gen):
    return [t async for t in gen]


def test_interpret_stream(interpreter):
    out = asyncio.run(_collect(interpreter.interpret_stream(["Pataka"], "arm raised")))
    assert out == ["a", "b"]
    call = interpreter.completions.calls[0]
    assert call["model"] == interp_mod.LLM_MODEL
    assert call["max_tokens"] == 1024 and call["stream"] is True
    assert call["reasoning_effort"] == "low"
    assert call["messages"][0] == {"role": "system", "content": interp_mod.SYSTEM_PROMPT}
    assert call["messages"][1]["content"] == (
        "Detected mudra(s): Pataka\nContext: arm raised\n\n"
        "Retrieved canonical viniyogas:\n- Pataka: cloud (raised)\n\n"
        "What is the dancer most likely expressing?"
    )


def test_interpret_stream_no_rag(interpreter):
    out = asyncio.run(_collect(interpreter.interpret_stream_no_rag(["Pataka"])))
    assert out == ["a", "b"]
    call = interpreter.completions.calls[0]
    assert call["messages"][0]["content"] == interp_mod.NO_RAG_SYSTEM_PROMPT
    assert call["messages"][1]["content"] == (
        "Detected mudra(s): Pataka\n\nWhat is the dancer most likely expressing?"
    )
    assert call["max_tokens"] == 1024 and call["stream"] is True


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        interp_mod.ViniyogaInterpreter()
