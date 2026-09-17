import unittest

from rag.chunker import chunk_pages
from rag.ingest import Page


class TestChunker(unittest.TestCase):
    def test_respects_chunk_size_and_overlap(self):
        # 100 distinct words on one page.
        words = [f"word{i}" for i in range(100)]
        page = Page(number=1, text=" ".join(words))

        chunks = chunk_pages([page], chunk_size=30, overlap=10)

        # stride = 20, so starts are 0, 20, 40, 60, 80 -> 5 chunks
        self.assertEqual(len(chunks), 5)
        for c in chunks[:-1]:
            self.assertEqual(len(c.text.split()), 30)
        # last chunk covers the remaining tail and may be shorter
        self.assertLessEqual(len(chunks[-1].text.split()), 30)

        # Verify overlap: last 10 words of chunk 0 == first 10 words of chunk 1
        overlap_from_chunk0 = chunks[0].text.split()[-10:]
        overlap_from_chunk1 = chunks[1].text.split()[:10]
        self.assertEqual(overlap_from_chunk0, overlap_from_chunk1)

    def test_chunk_ids_are_unique_and_sequential(self):
        words = [f"w{i}" for i in range(50)]
        page = Page(number=1, text=" ".join(words))
        chunks = chunk_pages([page], chunk_size=10, overlap=2)
        self.assertEqual([c.id for c in chunks], list(range(len(chunks))))

    def test_chunk_page_metadata_is_correct_across_pages(self):
        page1 = Page(number=1, text=" ".join(f"a{i}" for i in range(20)))
        page2 = Page(number=2, text=" ".join(f"b{i}" for i in range(20)))
        page3 = Page(number=3, text=" ".join(f"c{i}" for i in range(20)))

        chunks = chunk_pages([page1, page2, page3], chunk_size=10, overlap=2)

        for c in chunks:
            if c.text.startswith("a"):
                self.assertEqual(c.page, 1)
            elif c.text.startswith("b"):
                self.assertEqual(c.page, 2)
            elif c.text.startswith("c"):
                self.assertEqual(c.page, 3)
            else:
                self.fail(f"unexpected chunk text: {c.text!r}")

        # No chunk mixes words from two different pages.
        for c in chunks:
            prefixes = {w[0] for w in c.text.split()}
            self.assertEqual(len(prefixes), 1)

    def test_empty_page_produces_no_chunks(self):
        page = Page(number=1, text="")
        chunks = chunk_pages([page], chunk_size=10, overlap=2)
        self.assertEqual(chunks, [])

    def test_invalid_overlap_raises(self):
        page = Page(number=1, text="hello world")
        with self.assertRaises(ValueError):
            chunk_pages([page], chunk_size=10, overlap=10)
        with self.assertRaises(ValueError):
            chunk_pages([page], chunk_size=10, overlap=-1)


if __name__ == "__main__":
    unittest.main()
