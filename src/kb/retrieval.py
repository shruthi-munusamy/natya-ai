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
import numpy as np
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2


COLLECTION_NAME = "viniyoga"
DB_PATH = "data/kb/chroma"


class MiniLMEmbedder(ONNXMiniLM_L6_V2):
    """all-MiniLM-L6-v2 via Chroma's bundled ONNX export instead of
    sentence-transformers. Same model, same vectors (cosine 1.000000 against
    sentence-transformers on the same text), but no torch: the torch-backed
    version blew past Render's 512MB free-tier memory limit at startup.

    Model files live under the project (gitignored) rather than ~/.cache, so
    the Render build step can download them once and the running service
    finds them instead of re-downloading ~80MB on every cold start.
    """
    DOWNLOAD_PATH = Path(__file__).resolve().parents[2] / "data" / "kb" / "onnx_models" / "all-MiniLM-L6-v2"

    # Chroma's own forward pass pads every text to 256 tokens and embeds 32
    # at a time, which peaked near 700MB. Our texts are a sentence or two, so
    # pad each small batch only to its longest text. Padding is masked out of
    # both attention and mean pooling, so the vectors don't change; the
    # session itself (~165MB) is the floor either way.
    def _forward(self, documents, batch_size: int = 8):
        out = []
        for i in range(0, len(documents), batch_size):
            encoded = self.tokenizer.encode_batch(documents[i:i + batch_size])
            longest = max(len(e.ids) - e.attention_mask.count(0) for e in encoded)
            ids = np.array([e.ids[:longest] for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask[:longest] for e in encoded], dtype=np.int64)
            hidden = self.model.run(None, {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": np.zeros_like(ids),
            })[0]
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out.append(self._normalize(pooled).astype(np.float32))
        return np.concatenate(out)


class ViniyogaRetriever:
    def __init__(self, kb_path: str = "data/kb/viniyoga.json",
                 db_path: str = DB_PATH):
        self.kb_path = Path(kb_path)
        self.embedder = MiniLMEmbedder()
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

        embeddings = [e.tolist() for e in self.embedder(docs)]
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

        query_emb = [e.tolist() for e in self.embedder([query])]
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
    for hit in r.retrieve(["Pataka"], k=3):
        print(f"  - {hit['document']}  (dist={hit['distance']:.3f})")

    print("\nProbe: 'Pataka' with arm-raised context")
    for hit in r.retrieve(["Pataka"], context="arm raised above head", k=3):
        print(f"  - {hit['document']}  (dist={hit['distance']:.3f})")


if __name__ == "__main__":
    main()
