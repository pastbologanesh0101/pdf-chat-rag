"""Chunking: split per-page text into overlapping, page-tagged chunks.

Second stage of the RAG pipeline. Chunking happens *within* a single page's
text (chunks never span two pages), so every chunk carries an unambiguous,
correct page-number citation. Chunk size and overlap are measured in words
and are both configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rag.ingest import Page


@dataclass
class Chunk:
    """A chunk of text carrying provenance metadata for citation."""

    id: int
    page: int
    text: str
    word_start: int  # index of the first word of this chunk within its page's text
    word_end: int  # index one past the last word of this chunk within its page's text


def chunk_pages(pages: List[Page], chunk_size: int = 120, overlap: int = 30) -> List[Chunk]:
    """Split pages of text into overlapping word-window chunks.

    Args:
        pages: list of Page(number, text) from rag.ingest.extract_pages.
        chunk_size: maximum number of words per chunk. Must be >= 1.
        overlap: number of words shared between consecutive chunks on the
            same page. Must be >= 0 and < chunk_size.

    Returns:
        A list of Chunk objects with globally unique, sequential ids
        (starting at 0), each tagged with the 1-indexed source page number.
        Pages with empty/whitespace-only text produce no chunks.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Chunk] = []
    next_id = 0
    stride = chunk_size - overlap

    for page in pages:
        words = page.text.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                Chunk(
                    id=next_id,
                    page=page.number,
                    text=chunk_text,
                    word_start=start,
                    word_end=end,
                )
            )
            next_id += 1
            if end == len(words):
                break
            start += stride

    return chunks
