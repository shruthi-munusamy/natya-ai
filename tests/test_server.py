import importlib
import json

import pytest
from fastapi.testclient import TestClient


class _StubCollection:
    def count(self):
        return 1


class _StubRetriever:
    collection = _StubCollection()


class _StubInterpreter:
    def __init__(self):
        self.retriever = _StubRetriever()

    async def interpret_stream(self, mudras, context=None):
        for token in ["Pataka", " means", " cloud"]:
            yield token


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    import src.llm.interpreter
    monkeypatch.setattr(src.llm.interpreter, "ViniyogaInterpreter", _StubInterpreter)
    import src.llm.server
    server = importlib.reload(src.llm.server)
    with TestClient(server.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_interpret_stream_emits_tokens_then_done(client):
    res = client.post("/interpret/stream", json={"mudras": ["Pataka"], "context": None})
    assert res.status_code == 200
    events = [line.split(":", 1)[1].strip() for line in res.text.splitlines()
              if line.startswith("event:")]
    assert events == ["token", "token", "token", "done"]
    assert "data:  means" in res.text
