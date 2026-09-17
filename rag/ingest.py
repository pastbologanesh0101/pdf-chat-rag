"""PDF ingestion: extract per-page plain text from a PDF file.

This is the first stage of the RAG pipeline: ingest -> chunk -> index -> retrieve -> synthesize.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class Page:
    """A single extracted PDF page."""

    number: int  # 1-indexed page number, matches how a human would cite the PDF
    text: str


def extract_pages(pdf_path: str | Path) -> List[Page]:
    """Extract text from every page of a PDF.

    Args:
        pdf_path: path to a .pdf file.

    Returns:
        A list of Page objects, one per page, in order, 1-indexed by page number.
        Pages with no extractable text still appear in the list with text="".

    Raises:
        FileNotFoundError: if pdf_path does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: List[Page] = []
    for i, raw_page in enumerate(reader.pages, start=1):
        text = raw_page.extract_text() or ""
        pages.append(Page(number=i, text=text.strip()))
    return pages


def page_count(pdf_path: str | Path) -> int:
    """Return the number of pages in a PDF without extracting text."""
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)
