"""AI PDF Chat (RAG) - a fully local, no-LLM-API retrieval-augmented pipeline.

Modules:
    ingest      - extract per-page text from a PDF
    chunker     - split page text into overlapping, page-tagged chunks
    index       - TF-IDF vector index + cosine similarity search over chunks
    synthesize  - extractive "answer" from top retrieved chunks, with citations
"""

__version__ = "1.0.0"
