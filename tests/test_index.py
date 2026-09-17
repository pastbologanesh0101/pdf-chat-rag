import unittest

from rag.chunker import Chunk
from rag.index import TfidfIndex
from tests.fixtures import CRAFTED_DOCUMENT_PAGES


def _crafted_chunks():
    """One chunk per page of the crafted volcano/coral/glacier document."""
    return [
        Chunk(id=0, page=1, text=CRAFTED_DOCUMENT_PAGES[0], word_start=0, word_end=100),
        Chunk(id=1, page=2, text=CRAFTED_DOCUMENT_PAGES[1], word_start=0, word_end=100),
        Chunk(id=2, page=3, text=CRAFTED_DOCUMENT_PAGES[2], word_start=0, word_end=100),
    ]


class TestIndex(unittest.TestCase):
    def test_retrieves_the_genuinely_most_relevant_chunk(self):
        chunks = _crafted_chunks()
        index = TfidfIndex(chunks)

        results = index.search("What ecosystem is the Great Barrier Reef part of?", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.page, 2)
        self.assertIn("Coral reefs", results[0].chunk.text)

    def test_different_queries_retrieve_different_chunks(self):
        chunks = _crafted_chunks()
        index = TfidfIndex(chunks)

        volcano_result = index.search("How do shield volcanoes form from lava?", top_k=1)[0]
        glacier_result = index.search("How does a glacier carve a U-shaped valley?", top_k=1)[0]

        self.assertEqual(volcano_result.chunk.page, 1)
        self.assertEqual(glacier_result.chunk.page, 3)

    def test_top_k_respected_and_scores_descending(self):
        chunks = _crafted_chunks()
        index = TfidfIndex(chunks)
        results = index.search("coral reef ecosystem", top_k=2)
        self.assertLessEqual(len(results), 2)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_query_with_no_relevant_match_returns_empty(self):
        chunks = _crafted_chunks()
        index = TfidfIndex(chunks)
        results = index.search("quantum computing cryptography blockchain", top_k=3)
        self.assertEqual(results, [])

    def test_empty_index_returns_empty(self):
        index = TfidfIndex([])
        self.assertEqual(index.search("anything", top_k=3), [])


if __name__ == "__main__":
    unittest.main()
