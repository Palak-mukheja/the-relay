
import time
import unittest

from fastapi.testclient import TestClient
from src.standin import app, state


class TestStandin(unittest.TestCase):

    def setUp(self):

        self.client = TestClient(app)

        self.client.patch(
            "/admin/config",
            json={
                "mode": "normal",
                "rpm_limit": 600,
                "tpm_limit": 160000,
                "daily_token_budget": 4000000,
                "max_concurrency": 100,
                "slow_delay_ms": 0,
                "failure_probability": 0,
                "base_ttft_ms": 0,
                "base_total_ms": 8
            }
        )

        self.client.post("/admin/reset", json={"seed": 42})

        self.payload = {
            "model": "openai/gpt-oss-120b",
            "messages": [
                {"role": "user", "content": "Test question"}
            ],
            "stream": True,
            "max_completion_tokens": 32
        }

    def test_health_endpoint(self):

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "running")

    def test_streaming_response(self):

        response = self.client.post(
            "/v1/chat/completions",
            json=self.payload
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("[C1]", response.text)
        self.assertIn("[DONE]", response.text)
        self.assertIn('"usage"', response.text)

    def test_simulated_failure(self):

        self.client.patch(
            "/admin/config",
            json={"mode": "fail"}
        )

        response = self.client.post(
            "/v1/chat/completions",
            json=self.payload
        )

        self.assertEqual(response.status_code, 503)

    def test_rate_limit_and_retry_after(self):

        # Set the request limit to one per minute.
        self.client.patch(
            "/admin/config",
            json={"rpm_limit": 1}
        )

        # Check the server state before sending requests.
        state_before = self.client.get("/admin/state")
        print("\nSTATE BEFORE:", state_before.json())

        # First request should be accepted.
        first = self.client.post(
            "/v1/chat/completions",
            json=self.payload
        )

        print("\nFIRST RESPONSE:", first.status_code)
        print("FIRST BODY:", first.text)
        print("FIRST HEADERS:", dict(first.headers))

        # Second request should be rate-limited.
        second = self.client.post(
            "/v1/chat/completions",
            json=self.payload
        )

        print("\nSECOND RESPONSE:", second.status_code)
        print("SECOND BODY:", second.text)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertIn("Retry-After", second.headers)

    def test_daily_budget_exhaustion(self):

        self.client.patch(
            "/admin/config",
            json={
                "mode": "normal",
                "rpm_limit": 600,
                "tpm_limit": 160000,
                "daily_token_budget": 25,
                "base_ttft_ms": 0,
                "base_total_ms": 8
            }
        )

        self.client.post(
            "/admin/reset",
            json={"seed": 42}
        )

        first = self.client.post(
            "/v1/chat/completions",
            json=self.payload
        )

        self.assertEqual(first.status_code, 200)

        second = self.client.post(
            "/v1/chat/completions",
            json=self.payload
        )

        self.assertEqual(second.status_code, 429)

        body = second.json()

        self.assertEqual(
            body["error"]["code"],
            "daily_budget_exhausted"
        )

    def test_load_dependent_slowdown(self):

        state.update_config({
            "mode": "normal",
            "base_ttft_ms": 5,
            "base_total_ms": 15,
            "load_slowdown_threshold": 0,
            "load_slowdown_ms": 80
        })

        state.reset(seed=42)

        admission = state.admit(10)

        self.assertTrue(admission["allowed"])

        try:
            ttft, total = state.sample_latency()
        finally:
            state.release()

        self.assertGreaterEqual(ttft, 85)
        self.assertGreaterEqual(total, 95)
