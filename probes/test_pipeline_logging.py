import csv
import tempfile
import unittest
from pathlib import Path

from src.logger import Logger
from src.rag_core import Chunk
from src.pipeline_logging import run_and_log


class FakeRetriever:
    def search(self, query, k):
        chunk = Chunk(
            cid=0,
            paper="Moshi",
            page_start=1,
            page_end=1,
            text="This paper explains the model architecture."
        )
        return [(chunk, 1.0)]


class TestPipelineLogging(unittest.TestCase):

    def test_request_is_written_to_csv(self):

        def fake_llm(messages):
            return {
                "text": "The paper explains the model architecture [C1].",
                "tokens_in": 20,
                "tokens_out": 10,
                "first_token_ms": 5.0,
                "total_ms": 15.0
            }

        with tempfile.TemporaryDirectory() as temp_dir:

            csv_path = Path(temp_dir) / "test_results.csv"
            logger = Logger(csv_path)

            try:
                result = run_and_log(
                    question="What does the paper explain?",
                    retriever=FakeRetriever(),
                    llm=fake_llm,
                    logger=logger,
                    scenario="test",
                    model="fake-model",
                    retrieval_setting="M"
                )
            finally:
                logger.close()

            # Read the CSV we just created.
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(rows[0]["reply_kind"], "answered")
            self.assertEqual(rows[0]["source"], "real")
            self.assertTrue(rows[0]["first_token_ms"])
            self.assertTrue(rows[0]["work_ms"])
            self.assertEqual(rows[0]["queue_wait_ms"], "0.0")
            self.assertEqual(rows[0]["scenario"], "test")
            self.assertEqual(rows[0]["model"], "fake-model")

            self.assertEqual(rows[0]["tokens_in"], "20")
            self.assertEqual(rows[0]["tokens_out"], "10")
            self.assertEqual(rows[0]["retrieval_setting"], "M")

            self.assertTrue(rows[0]["request_id"])
            self.assertTrue(rows[0]["start_time"])


if __name__ == "__main__":
    unittest.main()
