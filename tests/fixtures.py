"""Shared test fixtures: build small, deterministic PDFs in a temp dir."""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a PDF where each element of `pages` becomes one page of text."""
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    for page_text in pages:
        y = height - 1 * inch
        c.setFont("Helvetica", 11)
        for word_group_start in range(0, len(page_text), 90):
            line = page_text[word_group_start : word_group_start + 90]
            c.drawString(1 * inch, y, line)
            y -= 16
        c.showPage()
    c.save()
    return path


def make_empty_pdf(path: Path) -> Path:
    """Write a PDF with a single blank page (no text at all)."""
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.showPage()
    c.save()
    return path


# A deterministic three-page document used for retrieval-accuracy tests.
# Page 1 is about volcanoes, page 2 is about coral reefs, page 3 is about
# glaciers. A query about coral reefs should retrieve page 2 above all else.
CRAFTED_DOCUMENT_PAGES = [
    (
        "Volcanoes are openings in the crust of a planet that allow molten "
        "rock, ash, and gases to escape from below the surface. Shield "
        "volcanoes like Mauna Loa form from very fluid lava that travels far "
        "before cooling, producing broad, gently sloping mountains."
    ),
    (
        "Coral reefs are underwater ecosystems built from calcium carbonate "
        "structures secreted by coral polyps. The Great Barrier Reef off the "
        "coast of Australia is the largest coral reef system in the world "
        "and hosts an enormous diversity of fish and marine invertebrates."
    ),
    (
        "Glaciers are large, slow-moving masses of ice formed from "
        "compacted snow over many years. As a glacier moves down a valley "
        "it carves out distinctive U-shaped valleys and can carry rock "
        "debris for long distances before depositing it as moraine."
    ),
]
