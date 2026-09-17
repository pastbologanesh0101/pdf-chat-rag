import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag.ingest import extract_pages, page_count
from tests.fixtures import CRAFTED_DOCUMENT_PAGES, make_empty_pdf, make_pdf


class TestIngest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_extraction_produces_nonempty_text_and_correct_page_count(self):
        pdf_path = make_pdf(self.tmp_path / "doc.pdf", CRAFTED_DOCUMENT_PAGES)
        pages = extract_pages(pdf_path)

        self.assertEqual(len(pages), 3)
        self.assertEqual(page_count(pdf_path), 3)
        for page in pages:
            self.assertTrue(len(page.text) > 0, f"page {page.number} extracted empty text")

    def test_page_numbers_are_1_indexed_and_sequential(self):
        pdf_path = make_pdf(self.tmp_path / "doc.pdf", CRAFTED_DOCUMENT_PAGES)
        pages = extract_pages(pdf_path)
        self.assertEqual([p.number for p in pages], [1, 2, 3])

    def test_extracted_text_contains_expected_content(self):
        pdf_path = make_pdf(self.tmp_path / "doc.pdf", CRAFTED_DOCUMENT_PAGES)
        pages = extract_pages(pdf_path)
        self.assertIn("Coral reefs", pages[1].text)
        self.assertIn("Glaciers", pages[2].text)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            extract_pages(self.tmp_path / "does_not_exist.pdf")

    def test_empty_pdf_handled_gracefully(self):
        pdf_path = make_empty_pdf(self.tmp_path / "empty.pdf")
        pages = extract_pages(pdf_path)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].text, "")


if __name__ == "__main__":
    unittest.main()
