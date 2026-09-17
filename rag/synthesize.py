"""Answer synthesis: extractive "answer" over retrieved chunks, with citations.

Fifth stage of the RAG pipeline, and the stage that normally calls an LLM.

=== Where a real LLM call would go ===
In a full generative RAG system, `synthesize_answer()` below is exactly the
function you would replace with something like:

    prompt = build_prompt(query, retrieved_chunks)
    answer = anthropic_client.messages.create(
        model="claude-...",
        messages=[{"role": "user", "content": prompt}],
    )

i.e. you would hand the query plus the retrieved chunk texts (with their
page numbers) to an LLM and ask it to write a fluent answer grounded in
those chunks, still citing the page numbers. Everything upstream of this
function (ingest -> chunk -> index -> retrieve) is unchanged either way —
that upstream pipeline *is* the "retrieval-augmented" part of RAG, and it
is fully implemented, tested, and running locally with no API key here.

This module instead performs *extractive* synthesis: it picks the
sentence(s) within the retrieved chunks that most overlap with the query
(via the same TF-IDF + cosine-similarity approach used for retrieval) and
returns them verbatim with a citation. No text is generated; every word of
the "answer" appears in the source PDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.index import RetrievedChunk

NO_MATCH_MESSAGE = "No relevant content found for this query in the document."

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Citation:
    page: int
    chunk_id: int

    def format(self) -> str:
        return f"(page {self.page}, chunk {self.chunk_id})"


@dataclass
class Answer:
    text: str
    citations: List[Citation]
    found: bool

    def format(self) -> str:
        if not self.found:
            return self.text
        cite_str = "; ".join(c.format() for c in self.citations)
        return f"{self.text} {cite_str}"


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return parts if parts else [text]


def synthesize_answer(
    query: str, retrieved: List[RetrievedChunk], max_sentences: int = 1
) -> Answer:
    """Produce an extractive answer + citation(s) from retrieved chunks.

    Args:
        query: the user's question.
        retrieved: ranked chunks from TfidfIndex.search (best first).
        max_sentences: how many top-scoring sentences to include in the answer.

    Returns:
        An Answer. If no retrieved chunks contain any sentence with
        non-zero relevance to the query, `found` is False and `text` is
        NO_MATCH_MESSAGE rather than a misleading guess.
    """
    if not retrieved:
        return Answer(text=NO_MATCH_MESSAGE, citations=[], found=False)

    sentences: List[str] = []
    sentence_sources: List[RetrievedChunk] = []
    for r in retrieved:
        for sentence in _split_sentences(r.chunk.text):
            sentences.append(sentence)
            sentence_sources.append(r)

    if not sentences:
        return Answer(text=NO_MATCH_MESSAGE, citations=[], found=False)

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        corpus = sentences + [query]
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Empty vocabulary (e.g. all stopwords / punctuation only).
        return Answer(text=NO_MATCH_MESSAGE, citations=[], found=False)

    query_vec = matrix[-1]
    sentence_matrix = matrix[:-1]
    scores = cosine_similarity(query_vec, sentence_matrix)[0]

    ranked_idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    top_idx = [i for i in ranked_idx if scores[i] > 1e-9][:max_sentences]

    if not top_idx:
        return Answer(text=NO_MATCH_MESSAGE, citations=[], found=False)

    top_idx_in_order = sorted(top_idx)
    answer_text = " ".join(sentences[i] for i in top_idx_in_order)
    citations = [
        Citation(page=sentence_sources[i].chunk.page, chunk_id=sentence_sources[i].chunk.id)
        for i in top_idx_in_order
    ]

    return Answer(text=answer_text, citations=citations, found=True)
