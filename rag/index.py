"""Embedding/retrieval: TF-IDF vector index over chunks with cosine similarity search.

Third and fourth stages of the RAG pipeline. This stands in for a dense
"embedding" model: each chunk is embedded as a sparse TF-IDF vector, and
retrieval is nearest-neighbor search under cosine similarity. It is a real,
well-understood retrieval method (not a stub) — it is simply not a neural
embedding model, and no LLM API is called anywhere in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.chunker import Chunk


@dataclass
class RetrievedChunk:
    """A chunk returned from a similarity search, with its score."""

    chunk: Chunk
    score: float


class TfidfIndex:
    """A TF-IDF vector index over a fixed set of chunks."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english")
        if chunks:
            texts = [c.text for c in chunks]
            self._matrix = self._vectorizer.fit_transform(texts)
        else:
            self._matrix = None

    def search(self, query: str, top_k: int = 3, min_score: float = 1e-9) -> List[RetrievedChunk]:
        """Return the top-k chunks most similar to `query`, best first.

        Chunks with a cosine similarity at or below `min_score` (i.e.
        effectively zero overlap with the query vocabulary) are excluded,
        since a near-zero score means "not actually relevant" rather than
        a genuine match.
        """
        if not self.chunks or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        ranked = sorted(
            (RetrievedChunk(chunk=c, score=float(s)) for c, s in zip(self.chunks, scores)),
            key=lambda r: r.score,
            reverse=True,
        )
        relevant = [r for r in ranked if r.score > min_score]
        return relevant[:top_k]
