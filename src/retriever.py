"""
retriever.py — Hybrid Retrieval: BM25 + Semantic Search with RRF Fusion
Uses FAISS instead of ChromaDB (no C++ build tools required on Windows)
"""

import json
import math
import os
import pickle
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Resolve paths relative to this file ───────────────────────────────────────
_SRC_DIR        = Path(__file__).parent          # …/bis-rag/src/
_ROOT_DIR       = _SRC_DIR.parent                # …/bis-rag/
CHUNKS_PATH     = str(_ROOT_DIR / "data" / "chunks.json")
BM25_CACHE_PATH = str(_ROOT_DIR / "data" / "bm25_index.pkl")
FAISS_CACHE_PATH= str(_ROOT_DIR / "data" / "faiss_index.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K      = 60

_model       = None
_faiss_index = None
_bm25_index  = None


# ── Model caching ──────────────────────────────────────────────────────────────

def get_model():
    global _model
    if _model is None:
        print("Loading sentence-transformer model...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ── FAISS Index (replaces ChromaDB) ───────────────────────────────────────────

class FAISSIndex:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query_embedding, top_k=20):
        import faiss
        results = []
        scores = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)
        for rank in range(top_k):
            idx   = scores[1][0][rank]
            score = float(scores[0][0][rank])
            if idx < len(self.chunks):
                results.append((idx, score))
        return results


def get_collection():
    """Compatibility shim — returns the FAISS index (replaces get_collection)."""
    return get_faiss_index()


def get_faiss_index(chunks_path=CHUNKS_PATH):
    global _faiss_index
    if _faiss_index is not None:
        return _faiss_index

    if os.path.exists(FAISS_CACHE_PATH):
        print("Loading FAISS index from cache...")
        with open(FAISS_CACHE_PATH, "rb") as f:
            _faiss_index = pickle.load(f)
        print(f"FAISS index ready ({len(_faiss_index.chunks)} chunks).")
        return _faiss_index

    # Build from scratch
    import faiss
    with open(chunks_path, encoding="utf-8") as f:
        all_chunks = json.load(f)
    valid_chunks = [c for c in all_chunks if c.get("standard_id") != "Unknown"]
    print(f"Building FAISS index over {len(valid_chunks)} chunks...")

    model  = get_model()
    texts  = [c["text"] for c in valid_chunks]

    print("Encoding chunks (this takes ~1-2 minutes the first time)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = embeddings.astype(np.float32)

    # L2-normalise for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine after normalisation
    index.add(embeddings)

    _faiss_index        = FAISSIndex(valid_chunks)
    _faiss_index.index  = index
    _faiss_index.embeddings = embeddings

    with open(FAISS_CACHE_PATH, "wb") as f:
        pickle.dump(_faiss_index, f)
    print("FAISS index built and cached.")
    return _faiss_index


# ── BM25 Index ─────────────────────────────────────────────────────────────────

def tokenize(text):
    text = text.lower()
    text = re.sub(r'\bis[\s:\-]*(\d)', r'is \1', text)
    tokens = re.findall(r'is\s*\d[\w\-]*|\b\w+\b', text)
    return tokens


class BM25Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.k1     = k1
        self.b      = b
        self.chunks = chunks
        self.n      = len(chunks)
        self.tf     = []
        self.df     = defaultdict(int)
        self.doc_lens = []

        for chunk in chunks:
            tokens = tokenize(chunk["text"] + " " + chunk.get("title", ""))
            tf_doc = defaultdict(int)
            for t in tokens:
                tf_doc[t] += 1
            self.tf.append(tf_doc)
            self.doc_lens.append(len(tokens))
            for term in tf_doc:
                self.df[term] += 1

        self.avg_dl = sum(self.doc_lens) / self.n if self.n > 0 else 1
        self.all_chunks = chunks

    def score(self, query_tokens, doc_idx):
        score  = 0.0
        tf_doc = self.tf[doc_idx]
        dl     = self.doc_lens[doc_idx]
        for term in query_tokens:
            if term not in tf_doc:
                continue
            n_t     = self.df.get(term, 0)
            idf     = math.log((self.n - n_t + 0.5) / (n_t + 0.5) + 1)
            tf_val  = tf_doc[term]
            tf_norm = (tf_val * (self.k1 + 1)) / (
                tf_val + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            )
            score += idf * tf_norm
        return score

    def search(self, query, top_k=20):
        query_tokens = tokenize(query)
        scores = [(i, self.score(query_tokens, i)) for i in range(self.n)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def get_bm25_index(chunks_path=CHUNKS_PATH):
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index

    if os.path.exists(BM25_CACHE_PATH):
        print("Loading BM25 index from cache...")
        with open(BM25_CACHE_PATH, "rb") as f:
            _bm25_index = pickle.load(f)
        print(f"BM25 index ready ({_bm25_index.n} chunks).")
        return _bm25_index

    with open(chunks_path, encoding="utf-8") as f:
        all_chunks = json.load(f)
    valid_chunks = [c for c in all_chunks if c.get("standard_id") != "Unknown"]
    print(f"Building BM25 index over {len(valid_chunks)} chunks...")
    _bm25_index = BM25Index(valid_chunks)

    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump(_bm25_index, f)
    print("BM25 index built and cached.")
    return _bm25_index


# ── RRF Fusion ─────────────────────────────────────────────────────────────────

def rrf_fuse(ranked_lists, k=RRF_K):
    rrf_scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            rrf_scores[item_id] += 1.0 / (k + rank)
    return rrf_scores


# ── Main retrieve function ─────────────────────────────────────────────────────

def retrieve(query, top_k=5, chunks_path=CHUNKS_PATH):
    # ── BM25 retrieval ─────────────────────────────────────────────────────
    bm25      = get_bm25_index(chunks_path)
    bm25_hits = bm25.search(query, top_k=20)

    bm25_ranked    = []
    seen_bm25      = set()
    bm25_chunk_map = {}

    positive_hits = [(idx, sc) for idx, sc in bm25_hits if sc > 0]
    for idx, score in positive_hits:
        chunk  = bm25.all_chunks[idx]
        std_id = chunk["standard_id"]
        if std_id not in seen_bm25:
            seen_bm25.add(std_id)
            bm25_ranked.append(std_id)
            bm25_chunk_map[std_id] = chunk

    # ── Semantic retrieval via FAISS ───────────────────────────────────────
    model       = get_model()
    faiss_index = get_faiss_index(chunks_path)
    query_emb   = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    sem_hits    = faiss_index.search(query_emb, top_k=20)

    sem_ranked    = []
    seen_sem      = set()
    sem_chunk_map = {}

    for idx, score in sem_hits:
        chunk  = faiss_index.chunks[idx]
        std_id = chunk["standard_id"]
        if std_id not in seen_sem:
            seen_sem.add(std_id)
            sem_ranked.append(std_id)
            sem_chunk_map[std_id] = {
                "standard_id": std_id,
                "title":       chunk.get("title", ""),
                "text":        chunk.get("text", ""),
                "score":       score,
            }

    # ── RRF fusion ─────────────────────────────────────────────────────────
    rrf_scores  = rrf_fuse([bm25_ranked, sem_ranked])
    all_std_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for std_id in all_std_ids[:top_k]:
        chunk = sem_chunk_map.get(std_id) or bm25_chunk_map.get(std_id, {})
        results.append({
            "standard_id": std_id,
            "title":       chunk.get("title", ""),
            "text":        chunk.get("text", ""),
            "score":       round(rrf_scores[std_id], 6),
            "source":      "hybrid",
        })

    return results


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "ordinary Portland cement for construction",
        "steel bars for reinforced concrete",
        "sand for masonry mortar",
        "coarse aggregates for concrete mix",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retrieve(q, top_k=5)
        for r in results:
            print(f"  [{r['score']}] {r['standard_id']} — {r['title'][:55]}")