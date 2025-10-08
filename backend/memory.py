import os
import json
import datetime
from typing import List, Dict, Any, Optional, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional advanced DBs
try:
    import faiss  # from faiss-cpu
    VECTOR_DB_AVAILABLE = True
except Exception:
    faiss = None
    VECTOR_DB_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except Exception:
    chromadb = None
    CHROMA_AVAILABLE = False

MEM_DIR = "memories"
os.makedirs(MEM_DIR, exist_ok=True)


class MemoryManager:
    def __init__(self, use_vector_db: bool = False, db_type: str = "faiss",
                 embedder: Optional[Callable[[str], Any]] = None):
        self.use_vector_db = bool(use_vector_db)
        self.db_type = db_type.lower()
        self.embedder = embedder

        # Always initialize attributes to satisfy Pylance
        self.index: Optional[Any] = None
        self.collection: Optional[Any] = None
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self.dim: Optional[int] = None

        if self.use_vector_db:
            if self.db_type == "faiss" and VECTOR_DB_AVAILABLE:
                # Lazy init when first embedding arrives
                pass
            elif self.db_type == "chroma" and CHROMA_AVAILABLE:
                try:
                    self.collection = chromadb.Client().create_collection("agent_memories")  # type: ignore
                except Exception:
                    self.collection = None
                    self.use_vector_db = False
            else:
                print("[MemoryManager] Vector DB not available, falling back to TF-IDF.")
                self.use_vector_db = False

    # -------------------------
    # JSON Local Memory
    # -------------------------
    def _file(self, agent_name: str) -> str:
        safe = "".join(c for c in agent_name if c.isalnum() or c in (" ", "_", "-")).rstrip()
        return os.path.join(MEM_DIR, f"{safe}.json")

    def load(self, agent_name: str) -> List[Dict[str, Any]]:
        path = self._file(agent_name)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def save(self, agent_name: str, text: str) -> None:
        mems = self.load(agent_name)
        entry = {"text": text, "timestamp": datetime.datetime.utcnow().isoformat()}
        mems.append(entry)
        with open(self._file(agent_name), "w", encoding="utf-8") as f:
            json.dump(mems, f, indent=2, ensure_ascii=False)

        if self.use_vector_db and self.embedder:
            try:
                emb = np.asarray(self.embedder(text), dtype="float32")
            except Exception:
                return
            if emb.ndim != 1:
                emb = emb.reshape(-1)

            if self.db_type == "faiss" and VECTOR_DB_AVAILABLE:
                if self.index is None:
                    self.dim = int(emb.shape[0])
                    self.index = faiss.IndexFlatL2(self.dim)  # type: ignore
                self.index.add(np.expand_dims(emb, axis=0))  # type: ignore
                self.vectors.append(emb)
                self.metadata.append(entry)

            elif self.db_type == "chroma" and self.collection is not None:
                new_id = str(len(mems))
                self.collection.add(documents=[text], metadatas=[entry], ids=[new_id])  # type: ignore

    # -------------------------
    # Retrieval
    # -------------------------
    def retrieve(self, agent_name: str, query: str, top_k: int = 3) -> List[str]:
        if self.use_vector_db and self.embedder:
            try:
                q_emb = np.asarray(self.embedder(query), dtype="float32")
                if q_emb.ndim != 1:
                    q_emb = q_emb.reshape(-1)
            except Exception:
                q_emb = None

            if q_emb is not None and self.db_type == "faiss" and self.index is not None and len(self.vectors) > 0:
                D, I = self.index.search(np.expand_dims(q_emb, axis=0), top_k)  # type: ignore
                return [self.metadata[i]["text"] for i in I[0] if 0 <= i < len(self.metadata)]

            if q_emb is not None and self.db_type == "chroma" and self.collection is not None:
                results = self.collection.query(query_texts=[query], n_results=top_k)  # type: ignore
                docs = results.get("documents")
                if isinstance(docs, list) and docs and isinstance(docs[0], list):
                    return docs[0]

        # # Fallback: TF-IDF
        # mems = self.load(agent_name)
        # texts = [m.get("text", "") for m in mems]
        # if not texts:
        #     return []
        # try:
        #     tfidf_matrix = TfidfVectorizer().fit_transform([query] + texts)
        #     query_vec = np.asarray(tfidf_matrix[0:1].todense())
        #     mem_vecs = np.asarray(tfidf_matrix[1:].todense())
        #     sims = cosine_similarity(query_vec, mem_vecs).flatten()
        #     idx_sorted = sims.argsort()[::-1][:top_k]
        #     return [texts[i] for i in idx_sorted]
        # except Exception:
        #     return texts[-top_k:]
        # Fallback: TF-IDF
        mems = self.load(agent_name)
        texts = [m.get("text", "") for m in mems]
        if not texts:
            return []
        try:
            tfidf_matrix = TfidfVectorizer().fit_transform([query] + texts)
            dense_matrix = np.asarray(tfidf_matrix.todense()) # convert once to ndarray
            query_vec = dense_matrix[0:1]
            mem_vecs = dense_matrix[1:]
            sims = cosine_similarity(query_vec, mem_vecs).flatten()
            idx_sorted = sims.argsort()[::-1][:top_k]
            return [texts[i] for i in idx_sorted]
        except Exception:
            return texts[-top_k:]


    # -------------------------
    # Prune
    # -------------------------
    def prune_memories(self, agent_name: str, max_entries: int = 50) -> List[Dict[str, Any]]:
        mems = self.load(agent_name)
        if len(mems) <= max_entries:
            return mems
        scored = [(i, m, len(m.get("text", "")) + (i / len(mems))) for i, m in enumerate(mems)]
        scored.sort(key=lambda x: x[2], reverse=True)
        pruned = [m for _, m, _ in scored[:max_entries]]
        with open(self._file(agent_name), "w", encoding="utf-8") as f:
            json.dump(pruned, f, indent=2, ensure_ascii=False)
        return pruned

    # -------------------------
    # Link memories (A-Mem)
    # -------------------------
    def link_memories(self, agent_name: str) -> List[Dict[str, Any]]:
        mems = self.load(agent_name)
        if len(mems) < 2:
            return mems
        texts = [m.get("text", "") for m in mems]
        try:
            tfidf_matrix = TfidfVectorizer().fit_transform(texts)
            dense_matrix = np.asarray(tfidf_matrix.todense())
            sims = cosine_similarity(dense_matrix)
            links = []
            for i in range(len(mems)):
                related = [(j, sims[i][j]) for j in range(len(mems)) if i != j]
                related.sort(key=lambda x: x[1], reverse=True)
                top_related = [mems[j]["text"] for j, _ in related[:3]]
                links.append({"text": mems[i]["text"], "related": top_related})
            return links
        except Exception:
            return [{"text": m.get("text", ""), "related": []} for m in mems]
        return mems 
