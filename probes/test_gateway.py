
import asyncio
import time
import unittest

from src.gateway import Gateway


class TestGateway(unittest.TestCase):

    def test_successful_request(self):

        async def run():

            gateway = Gateway(
                max_concurrency=2,
                queue_size=5,
                tpm=60000,
                deadline_seconds=3
            )

            await gateway.start()

            try:
                result = await gateway.submit(
                    lambda: {
                        "status": "answered",
                        "tokens_in": 10,
                        "tokens_out": 5
                    },
                    estimated_tokens=20
                )

                self.assertEqual(result["infra_status"], "ok")
                self.assertEqual(result["status"], "answered")
                self.assertEqual(result["gateway_attempts"], 1)
                self.assertGreaterEqual(result["total_ms"], 0)

            finally:
                await gateway.close()

        asyncio.run(run())

    def test_queue_limit(self):

        async def run():

            gateway = Gateway(
                max_concurrency=1,
                queue_size=1,
                tpm=60000,
                deadline_seconds=3
            )

            await gateway.start()

            try:

                # This request occupies the worker.
                first = asyncio.create_task(
                    gateway.submit(
                        lambda: (time.sleep(0.3) or {
                            "status": "answered",
                            "tokens_in": 5,
                            "tokens_out": 5
                        }),
                        estimated_tokens=10
                    )
                )

                await asyncio.sleep(0.05)

                # This request fills the waiting queue.
                second = asyncio.create_task(
                    gateway.submit(
                        lambda: {
                            "status": "answered",
                            "tokens_in": 5,
                            "tokens_out": 5
                        },
                        estimated_tokens=10
                    )
                )

                await asyncio.sleep(0.05)

                # This request should be rejected.
                third = await gateway.submit(
                    lambda: {
                        "status": "answered",
                        "tokens_in": 5,
                        "tokens_out": 5
                    },
                    estimated_tokens=10
                )

                self.assertEqual(
                    third["infra_status"],
                    "timeout"
                )

                await first
                await second

            finally:
                await gateway.close()

        asyncio.run(run())

    def test_retry_budget_and_retry_after(self):

        async def run():

            gateway = Gateway(
                max_concurrency=1,
                queue_size=5,
                tpm=60000,
                deadline_seconds=3,
                retry_budget_fraction=1.0,
                max_retries=1,
                base_backoff=0.01,
                jitter=0
            )

            attempts = {"count": 0}

            def sometimes_fails():

                attempts["count"] += 1

                if attempts["count"] == 1:
                    return {
                        "status": "error",
                        "error_message": "HTTP 429 retry-after=0.01",
                        "tokens_in": 5,
                        "tokens_out": 0
                    }

                return {
                    "status": "answered",
                    "tokens_in": 5,
                    "tokens_out": 5
                }

            await gateway.start()

            try:
                result = await gateway.submit(
                    sometimes_fails,
                    estimated_tokens=10
                )

                self.assertEqual(result["infra_status"], "ok")
                self.assertEqual(result["gateway_attempts"], 2)
                self.assertEqual(result["gateway_retries"], 1)
                self.assertGreater(result["retry_wait_ms"], 0)

            finally:
                await gateway.close()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
