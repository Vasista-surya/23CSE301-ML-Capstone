"""Smoke tests for the Vercel serverless adapter."""

from __future__ import annotations

import unittest

from api.index import app


class VercelApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = app.test_client()

    def test_health_route(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_summary_route(self) -> None:
        response = self.client.get("/api/summary")
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["datasets"]["fifa.csv"]["discoverable"])

    def test_bounded_preview_route(self) -> None:
        response = self.client.get("/api/preview?dataset=nutrition&n_rows=2")
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["data"]["rows"]), 2)


if __name__ == "__main__":
    unittest.main()
