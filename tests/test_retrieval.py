import subprocess
from pathlib import Path

import pytest

from src.kb.retrieval import ViniyogaRetriever

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("kb")
    kb = tmp / "viniyoga.json"
    kb.write_bytes(subprocess.check_output(
        ["git", "show", "HEAD:data/kb/viniyoga.json"], cwd=REPO
    ))
    r = ViniyogaRetriever(kb_path=str(kb), db_path=str(tmp / "chroma"))
    assert r.build_index() > 0
    return r


def test_retrieve_filters_to_requested_mudra(retriever):
    hits = retriever.retrieve(["Pataka"], k=3)
    assert 0 < len(hits) <= 3
    assert all(h["metadata"]["mudra"] == "Pataka" for h in hits)


def test_retrieve_with_context(retriever):
    hits = retriever.retrieve(["Pataka"], context="arm raised above head", k=3)
    assert hits and all(h["metadata"]["mudra"] == "Pataka" for h in hits)
