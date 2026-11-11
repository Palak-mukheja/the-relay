
import asyncio
import threading
import time
import unittest

from src.gateway import Gateway


class TestGatewayStress(unittest.TestCase):

    def test_concurrency_cap_with_simultaneous_requests(self):

        async def run():

            gateway = Gateway(
                max_concurrency=2,
                queue_size=10,
                tpm=60000,
                deadline_seconds=2
            )

            lock = threading.Lock()

            active = 0
            peak_active = 0

            def fake_service():

                nonlocal active, peak_active

                with lock:
                    active += 1
                    peak_active = max(peak_active, active)

                try:
                    time.sleep(0.1)

                    return {
                        "status": "answered",
                        "tokens_in": 20,
                        "tokens_out": 10
                    }

                finally:
                    with lock:
                        active -= 1

            await gateway.start()

            try:

                tasks = [
                    asyncio.create_task(
                        gateway.submit(
                            fake_service,
                            estimated_tokens=30
                        )
                    )
                    for _ in range(10)
                ]

                results = await asyncio.gather(*tasks)

                # Every request should succeed.
                self.assertEqual(len(results), 10)

                for result in results:
                    self.assertEqual(result["infra_status"], "ok")

                # Never exceed two simultaneous requests.
                self.assertLessEqual(peak_active, 2)

                # Verify that workers actually ran concurrently.
                self.assertEqual(peak_active, 2)

                # All requests should have been processed.
                self.assertEqual(
                    sum(r["gateway_attempts"] for r in results),
                    10
                )

                # Check the queue and timing measurements.
                self.assertGreater(gateway.queue_peak, 0)
                self.assertLessEqual(gateway.queue_peak, 10)

                for result in results:
                    calculated = (
                        result["queue_wait_ms"]
                        + result["work_ms"]
                        + result["retry_wait_ms"]
                    )

                    self.assertAlmostEqual(
                        result["total_ms"],
                        calculated,
                        places=3
                    )

            finally:
                await gateway.close()

        asyncio.run(run())

    def test_request_deadline_expires_while_queued(self):

        async def run():

            gateway = Gateway(
                max_concurrency=1,
                queue_size=5,
                tpm=60000,
                deadline_seconds=2
            )

            def slow_service():
                time.sleep(0.2)

                return {
                    "status": "answered",
                    "tokens_in": 10,
                    "tokens_out": 5
                }

            await gateway.start()

            try:

                first = asyncio.create_task(
                    gateway.submit(
                        slow_service,
                        estimated_tokens=15,
                        request_timeout=1
                    )
                )

                await asyncio.sleep(0.02)

                second = asyncio.create_task(
                    gateway.submit(
                        slow_service,
                        estimated_tokens=15,
                        request_timeout=0.05
                    )
                )

                first_result, second_result = await asyncio.gather(
                    first, second
                )

                self.assertEqual(first_result["infra_status"], "ok")
                self.assertEqual(second_result["infra_status"], "timeout")

            finally:
                await gateway.close()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
