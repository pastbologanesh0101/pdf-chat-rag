#!/usr/bin/env python3
"""AI PDF Chat (RAG) - CLI entry point.

Fully local retrieval-augmented pipeline over a PDF, with NO LLM API call:

    ingest -> chunk -> embed/index (TF-IDF) -> retrieve (cosine similarity) -> synthesize (extractive)

Usage:
    python chat.py document.pdf "What is the main conclusion?"
    python chat.py document.pdf --interactive
    python chat.py document.pdf "..." --top-k 5 --chunk-size 150 --overlap 40

See README.md for the full architecture explanation and where a real LLM
call would replace the extractive synthesis step.
"""

from __future__ import annotations

import argparse
import sys

from rag.chunker import chunk_pages
from rag.index import TfidfIndex
from rag.ingest import extract_pages
from rag.synthesize import synthesize_answer


def build_index(pdf_path: str, chunk_size: int, overlap: int) -> TfidfIndex:
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    return TfidfIndex(chunks)


def answer_query(index: TfidfIndex, query: str, top_k: int) -> str:
    retrieved = index.search(query, top_k=top_k)
    answer = synthesize_answer(query, retrieved)
    lines = [f"Q: {query}", f"A: {answer.format()}"]
    if retrieved:
        lines.append("")
        lines.append("Top retrieved chunks:")
        for r in retrieved:
            preview = r.chunk.text[:120] + ("..." if len(r.chunk.text) > 120 else "")
            lines.append(f"  [chunk {r.chunk.id} | page {r.chunk.page} | score {r.score:.3f}] {preview}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="AI PDF Chat (RAG): local, no-LLM-API retrieval over a PDF."
    )
    parser.add_argument("pdf_path", help="Path to the PDF to chat with.")
    parser.add_argument(
        "question", nargs="?", default=None, help="Question to ask (omit with --interactive)."
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3).")
    parser.add_argument(
        "--chunk-size", type=int, default=120, help="Words per chunk (default: 120)."
    )
    parser.add_argument(
        "--overlap", type=int, default=30, help="Overlapping words between chunks (default: 30)."
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Start a REPL for follow-up questions."
    )
    args = parser.parse_args(argv)

    try:
        index = build_index(args.pdf_path, args.chunk_size, args.overlap)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not index.chunks:
        print("Warning: no extractable text/chunks were found in this document.")
        if not args.interactive:
            return 0

    if args.interactive:
        print(f"Indexed {len(index.chunks)} chunks from {args.pdf_path}.")
        print("Interactive mode. Type a question, or 'quit' / 'exit' to leave.\n")
        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            if query.lower() in {"quit", "exit"}:
                break
            print(answer_query(index, query, args.top_k))
            print()
        return 0

    if not args.question:
        parser.error("a question is required unless --interactive is used")

    print(answer_query(index, args.question, args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
