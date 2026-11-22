"""API tests for the operator console server (server.py), with a stubbed pipeline."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TMP = tempfile.mkdtemp()
os.environ["SENTINEL_SKIP_WARMUP"] = "1"
os.environ["SENTINEL_FEEDBACK_FILE"] = os.path.join(TMP, "feedback.jsonl")

try:
    from starlette.testclient import TestClient

    import server
except ImportError as error:  # starlette / numpy not installed (bare CI)
    raise unittest.SkipTest(f"server dependencies unavailable: {error}")

FAKE_RESULT = {
    "type": "Request", "type_confidence": 0.95, "priority": "low", "priority_confidence": 0.9,
    "queue": "Billing and Payments", "queue_confidence": 0.9, "confidence": 0.92, "detected_language": "en",
    "escalation_status": "AUTO_DRAFT", "escalation_reason": "Passed safety gate", "escalation_triggers": [],
    "top_similarity": 0.81, "unsupported_requests": [],
    "retrieved_precedents": [{"ticket_id": "TKT-00042", "similarity": 0.81, "historical_answer": "Plans are on the pricing page.",
                              "subject": "Plans", "language": "en", "queue": "Billing and Payments", "type": "Request",
                              "priority": "low"}],
    "draft_response": "See the pricing page [TKT-00042].", "draft_source": "template", "draft_verified": True,
    "draft_issues": [], "draft_rejected": [], "llm_error": "", "pii_masked": {"EMAIL": 1},
    "type_ms": 10.0, "priority_ms": 11.0, "queue_ms": 1.0, "retrieval_ms": 50.0, "draft_ms": 2.0, "latency_ms": 64.0,
}


class FakeGraph:
    def __init__(self):
        self.calls = []

    def invoke(self, state, config=None):
        self.calls.append((state, config))
        return dict(FAKE_RESULT)


class ServerAPI(unittest.TestCase):
    def setUp(self):
        self.graph = FakeGraph()
        server.STATE.update({"graph": self.graph, "error": None})
        patches = [
            mock.patch.object(server, "queue_top_terms", return_value=["pricing"]),
            mock.patch.object(server.BURST, "add", return_value=None),  # avoid loading the embedding model
            mock.patch.dict(os.environ, {"GEMINI_API_KEY": "secret-test-key-123"}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(server.app)

    def test_index_and_health_never_expose_the_key(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        response = self.client.get("/api/health")
        self.assertTrue(response.json()["ready"])
        self.assertTrue(response.json()["gemini"])
        self.assertNotIn("secret-test-key", response.text)

    def test_triage_requires_a_message(self):
        self.assertEqual(self.client.post("/api/triage", json={"subject": "x", "body": "  "}).status_code, 400)

    def test_triage_returns_public_fields_and_highlights(self):
        response = self.client.post("/api/triage", json={"subject": "Plans", "body": "Mail me at a@b.com about plans",
                                                          "use_llm": False})
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "AUTO_DRAFT")
        self.assertEqual(data["queue_terms"], ["pricing"])
        self.assertEqual(data["precedents"][0]["ticket_id"], "TKT-00042")
        self.assertNotIn("type", data["precedents"][0])  # only display fields leave the server
        self.assertEqual([s["kind"] for s in data["highlights"]["body"]], ["pii"])
        self.assertNotIn("secret-test-key", response.text)
        # use_llm=False forces offline drafting through the per-call config
        self.assertEqual(self.graph.calls[-1][1], {"configurable": {"gemini_api_key": ""}})

    def test_feedback_is_masked_and_validated(self):
        self.assertEqual(self.client.post("/api/feedback", json={"action": "delete"}).status_code, 400)
        response = self.client.post("/api/feedback", json={
            "action": "correct", "subject": "Call +49 170 1234567", "body": "My email is jane@corp.com",
            "predicted": {"type": "Request"}, "correction": {"queue": "IT Support", "evil": "x"}, "draft": "ok"})
        self.assertTrue(response.json()["ok"])
        lines = Path(os.environ["SENTINEL_FEEDBACK_FILE"]).read_text().strip().splitlines()
        entry = json.loads(lines[-1])
        self.assertNotIn("jane@corp.com", entry["body"])
        self.assertNotIn("1234567", entry["subject"])
        self.assertEqual(entry["correction"], {"queue": "IT Support"})

    def test_stream_sends_only_subject_and_body(self):
        if not server.HOLDOUT_FILE.is_file():
            self.skipTest("holdout split not generated")
        rows = self.client.get("/api/stream?n=5").json()
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertEqual(set(row), {"id", "subject", "body"})

    def test_models_loading_returns_503(self):
        server.STATE.update({"graph": None, "error": None})
        self.assertEqual(self.client.post("/api/triage", json={"body": "hello"}).status_code, 503)


if __name__ == "__main__":
    unittest.main()
