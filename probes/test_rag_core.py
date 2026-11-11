import unittest
from src.rag_core import Chunk, make_chunks, answer, validate



class EmptyRetriever:
    def search(self, query, k):
        return []
class OneHitRetriever:
    def search(self, query, k):
        chunk = Chunk(
            cid=0,
            paper="Moshi",
            page_start=1,
            page_end=1,
            text="A passage containing some evidence."
        )
        return [(chunk, 1.0)]

class TestRAGCore(unittest.TestCase):
    def test_validator_accepts_fullwidth_citations(self):
        chunk = Chunk(
            cid=0,
            paper="Attention Is All You Need",
            page_start=4,
            page_end=5,
            text="Each attention head uses dk=64 and dv=64."
        )

        hits = [(chunk, 1.0)]

        status, citations, reasons = validate(
            "The query and key dimensions are 64, and the value "
            "dimension is 64 【C1】.",
            hits
        )

        self.assertEqual(status, "ok")
        self.assertEqual(len(citations), 1)
        self.assertEqual(reasons, [])

    def test_successful_request_records_metrics(self):

        def fake_llm(messages):
            return {
                "text": "The passage explains the model architecture [C1].",
                "tokens_in": 12,
                "tokens_out": 7,
                "first_token_ms": 5.0,
                "total_ms": 10.0
            }

        result = answer(
            question="What does the paper explain?",
            retriever=OneHitRetriever(),
            llm=fake_llm
        )

        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["tokens_in"], 12)
        self.assertEqual(result["tokens_out"], 7)
        self.assertGreaterEqual(result["first_token_ms"], 5.0)
        self.assertGreaterEqual(result["total_ms"], result["first_token_ms"])

    def test_model_failure_does_not_crash_pipeline(self):

        def failing_llm(messages):
            raise RuntimeError("HTTP 503: server overloaded")

        result = answer(
            question="What does the paper explain?",
            retriever=OneHitRetriever(),
            llm=failing_llm
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("HTTP 503", result["error_message"])
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["tokens_in"], 0)
        self.assertEqual(result["tokens_out"], 0)

    def test_chunks_do_not_cross_excluded_pages(self):
        papers = {
            "Moshi": [
                (1, "alpha alpha alpha alpha alpha alpha"),
                (2, "bravo bravo bravo bravo bravo bravo"),
                (3, "BIBLIOGRAPHYSECRET BIBLIOGRAPHYSECRET"),
                (4, "delta delta delta delta delta delta"),
                (5, "echo echo echo echo echo echo"),
            ]
        }

        chunks = make_chunks(
            papers,
            size_words=8,
            overlap_words=2,
            skip_pages={"Moshi": {3}}
        )

        self.assertTrue(chunks)

        for chunk in chunks:
            # No chunk should include the excluded page.
            self.assertFalse(
                chunk.page_start <= 3 <= chunk.page_end
            )

            # Excluded text must not appear in any chunk.
            self.assertNotIn(
                "BIBLIOGRAPHYSECRET",
                chunk.text
            )

            # No chunk should jump across the excluded page.
            self.assertFalse(
                chunk.page_start < 3 < chunk.page_end
            )

    def test_empty_retrieval_refuses_without_calling_llm(self):

        def fake_llm(messages):
            raise AssertionError("LLM should not have been called")

        result = answer(
            question="What is the answer to this question?",
            retriever=EmptyRetriever(),
            llm=fake_llm
        )

        self.assertEqual(result["status"], "refused")
        self.assertEqual(result["tokens_in"], 0)
        self.assertEqual(result["tokens_out"], 0)
        self.assertEqual(result["attempts"], 1)


if __name__ == "__main__":
    unittest.main()