"""Vector retrieval over the viniyoga knowledge base.

Each viniyoga entry (one mudra, one meaning, one context) becomes a separate
document in the index. At inference time, given a detected mudra (or set of
mudras + optional body pose context), we retrieve top-k most relevant entries
to ground the LLM's interpretation.

Why a vector store rather than a dict lookup: for v1 with single-mudra
queries it's overkill. But the LLM downstream benefits from semantic ranking
when we add sequence and pose context — 'Pataka with arm raised' should
retrieve cloud/sky viniyogas first, not all six equally. Embedding-based
retrieval makes that ranking principled.
"""
import json
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "viniyoga"
DB_PATH = "data/kb/chroma"


class ViniyogaRetriever:
    def __init__(self, kb_path: str = "data/kb/viniyoga.json",
                 db_path: str = DB_PATH):
        self.kb_path = Path(kb_path)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(self):
        """Rebuild the vector index from the KB JSON. Idempotent."""
        with open(self.kb_path) as f:
            kb = json.load(f)

        # Wipe and recreate to handle deletions cleanly
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        docs, ids, metadatas = [], [], []
        for mudra_name, mudra in kb["mudras"].items():
            for i, v in enumerate(mudra["viniyogas"]):
                doc = (
                    f"{mudra_name} ({mudra.get('translation', '')}): "
                    f"{v['meaning']}. {v['context']}"
                )
                docs.append(doc)
                ids.append(f"{mudra_name}__{i}")
                metadatas.append({
                    "mudra": mudra_name,
                    "type": mudra["type"],
                    "meaning": v["meaning"],
                    "context": v["context"],
                })

        embeddings = self.embedder.encode(docs, show_progress_bar=False).tolist()
        self.collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(docs)

    def retrieve(
        self,
        mudras: list[str],
        context: Optional[str] = None,
        k: int = 5,
    ) -> list[dict]:
        """Retrieve top-k viniyoga entries for the given mudras + optional context.

        mudras: list of detected mudra names (single hand or both)
        context: free-text context — pose info, sequence history, etc.
                 Used at inference time once the LLM layer is wired (C7).
        """
        query = ", ".join(mudras)
        if context:
            query = f"{query}. Context: {context}"

        query_emb = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=k,
            where={"mudra": {"$in": mudras}} if len(mudras) > 0 else None,
        )

        out = []
        for i in range(len(results["ids"][0])):
            out.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return out


def main():
    """Build index from CLI."""
    r = ViniyogaRetriever()
    n = r.build_index()
    print(f"Indexed {n} viniyoga entries.")

    # Sanity probe
    print("\nProbe: 'Pataka' alone")
    for hit in r.retrieve(["Pathaka"], k=3):
        print(f"  - {hit['document']}  (dist={hit['distance']:.3f})")

    print("\nProbe: 'Pataka' with arm-raised context")
    for hit in r.retrieve(["Pathaka"], context="arm raised above head", k=3):
        print(f"  - {hit['document']}  (dist={hit['distance']:.3f})")


if __name__ == "__main__":
    main()
