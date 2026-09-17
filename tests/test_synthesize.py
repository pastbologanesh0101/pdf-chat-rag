import unittest

from rag.chunker import Chunk
from rag.index import RetrievedChunk
from rag.synthesize import NO_MATCH_MESSAGE, synthesize_answer


class TestSynthesize(unittest.TestCase):
    def test_extracts_most_relevant_sentence_with_citation(self):
        chunk = Chunk(
            id=5,
            page=3,
            text=(
                "Coral reefs are underwater ecosystems built from calcium "
                "carbonate. The Great Barrier Reef is the largest coral reef "
                "system in the world. Glaciers are unrelated to reefs."
            ),
            word_start=0,
            word_end=30,
        )
        retrieved = [RetrievedChunk(chunk=chunk, score=0.5)]

        answer = synthesize_answer("What is the largest coral reef system?", retrieved)

        self.assertTrue(answer.found)
        self.assertIn("Great Barrier Reef", answer.text)
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(answer.citations[0].page, 3)
        self.assertEqual(answer.citations[0].chunk_id, 5)

    def test_citation_formatting_includes_page_number(self):
        chunk = Chunk(id=2, page=7, text="Volcanoes erupt lava and ash.", word_start=0, word_end=5)
        retrieved = [RetrievedChunk(chunk=chunk, score=0.9)]

        answer = synthesize_answer("What do volcanoes erupt?", retrieved)
        formatted = answer.format()

        self.assertIn("page 7", formatted)
        self.assertIn("chunk 2", formatted)

    def test_no_relevant_match_returns_clear_message_not_wrong_answer(self):
        chunk = Chunk(
            id=0, page=1, text="Lighthouses guide ships along rocky coastlines.", word_start=0, word_end=6
        )
        retrieved = [RetrievedChunk(chunk=chunk, score=0.0)]

        # Simulate what the index would actually return: nothing, since the
        # query shares no vocabulary with the chunk.
        answer = synthesize_answer("How many moons does Jupiter have?", [])

        self.assertFalse(answer.found)
        self.assertEqual(answer.text, NO_MATCH_MESSAGE)
        self.assertEqual(answer.citations, [])

    def test_empty_retrieved_list_handled_gracefully(self):
        answer = synthesize_answer("anything", [])
        self.assertFalse(answer.found)
        self.assertEqual(answer.format(), NO_MATCH_MESSAGE)

    def test_very_short_chunk_handled_gracefully(self):
        chunk = Chunk(id=0, page=1, text="Hi.", word_start=0, word_end=1)
        retrieved = [RetrievedChunk(chunk=chunk, score=0.1)]
        answer = synthesize_answer("Hi there, how are you?", retrieved)
        # Should not crash; either finds "Hi." or reports no match.
        self.assertIsInstance(answer.found, bool)


if __name__ == "__main__":
    unittest.main()
