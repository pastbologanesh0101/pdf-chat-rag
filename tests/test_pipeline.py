import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag.chunker import chunk_pages
from rag.index import TfidfIndex
from rag.ingest import extract_pages
from rag.synthesize import NO_MATCH_MESSAGE, synthesize_answer
from tests.fixtures import CRAFTED_DOCUMENT_PAGES, make_empty_pdf, make_pdf


class TestFullPipeline(unittest.TestCase):
    """End-to-end tests: real PDF -> extract -> chunk -> index -> synthesize."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_end_to_end_retrieval_and_citation_on_crafted_document(self):
        pdf_path = make_pdf(self.tmp_path / "doc.pdf", CRAFTED_DOCUMENT_PAGES)

        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages, chunk_size=200, overlap=20)
        index = TfidfIndex(chunks)

        retrieved = index.search("Tell me about coral reef ecosystems and the Great Barrier Reef", top_k=1)
        answer = synthesize_answer("Tell me about coral reef ecosystems and the Great Barrier Reef", retrieved)

        self.assertTrue(answer.found)
        self.assertIn("reef", answer.text.lower())
        self.assertEqual(answer.citations[0].page, 2)

    def test_empty_document_handled_gracefully_end_to_end(self):
        pdf_path = make_empty_pdf(self.tmp_path / "empty.pdf")

        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages, chunk_size=120, overlap=30)
        self.assertEqual(chunks, [])

        index = TfidfIndex(chunks)
        retrieved = index.search("anything at all", top_k=3)
        answer = synthesize_answer("anything at all", retrieved)

        self.assertFalse(answer.found)
        self.assertEqual(answer.text, NO_MATCH_MESSAGE)

    def test_very_short_document_handled_gracefully_end_to_end(self):
        pdf_path = make_pdf(self.tmp_path / "short.pdf", ["Hello."])

        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages, chunk_size=120, overlap=30)
        index = TfidfIndex(chunks)

        # No crash, and a reasonable outcome either way (no vocabulary
        # overlap or a match) for an unrelated query.
        retrieved = index.search("unrelated topic about spacecraft propulsion", top_k=3)
        answer = synthesize_answer("unrelated topic about spacecraft propulsion", retrieved)
        self.assertFalse(answer.found)


if __name__ == "__main__":
    unittest.main()
