"""Tests for the safety escalation gate and its wiring into the LangGraph workflow.

Run from the repo root:
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety_gate import (  # noqa: E402
    CLASSIFIER_UNAVAILABLE,
    HIGH_PRIORITY_INCIDENT,
    LOW_CONFIDENCE,
    MISSING_PREDICTION,
    evaluate_safety_gate,
)


def gate(**overrides):
    args = dict(
        ticket_type="Request",
        priority="low",
        confidence=0.90,
        unavailable_classifiers=(),
        threshold=0.65,
        escalate_on_fallback=False,
    )
    args.update(overrides)
    return evaluate_safety_gate(**args)


class SafetyGateRules(unittest.TestCase):
    def test_confident_routine_ticket_auto_drafts(self):
        decision = gate()
        self.assertEqual(decision.route, "auto_draft")
        self.assertEqual(decision.status, "AUTO_DRAFT")
        self.assertEqual(decision.triggers, ())

    def test_high_priority_incident_escalates_even_when_confident(self):
        decision = gate(ticket_type="Incident", priority="high", confidence=0.99)
        self.assertEqual(decision.status, "ESCALATE")
        self.assertEqual(decision.triggers, (HIGH_PRIORITY_INCIDENT,))

    def test_labels_are_case_and_whitespace_insensitive(self):
        decision = gate(ticket_type=" incident ", priority="HIGH")
        self.assertIn(HIGH_PRIORITY_INCIDENT, decision.triggers)

    def test_high_priority_non_incident_is_not_rule_one(self):
        self.assertEqual(gate(ticket_type="Request", priority="high").route, "auto_draft")

    def test_medium_incident_is_not_rule_one(self):
        self.assertEqual(gate(ticket_type="Incident", priority="medium").route, "auto_draft")

    def test_confidence_below_threshold_escalates(self):
        decision = gate(confidence=0.64)
        self.assertEqual(decision.triggers, (LOW_CONFIDENCE,))
        self.assertIn("0.64", decision.reason)

    def test_confidence_exactly_at_threshold_passes(self):
        self.assertEqual(gate(confidence=0.65).route, "auto_draft")

    def test_threshold_is_configurable(self):
        self.assertEqual(gate(confidence=0.68, threshold=0.70).route, "escalate")
        self.assertEqual(gate(confidence=0.68, threshold=0.65).route, "auto_draft")

    def test_both_rules_reported_together(self):
        decision = gate(ticket_type="Incident", priority="high", confidence=0.40)
        self.assertEqual(decision.triggers, (HIGH_PRIORITY_INCIDENT, LOW_CONFIDENCE))
        self.assertIn("High-Priority Incident", decision.reason)
        self.assertIn("below threshold", decision.reason)

    def test_missing_or_invalid_values_fail_safe(self):
        for bad in (None, float("nan"), -0.1, 1.5, "0.9", True):
            with self.subTest(confidence=bad):
                decision = gate(confidence=bad)
                self.assertEqual(decision.route, "escalate")
                self.assertIn(MISSING_PREDICTION, decision.triggers)
        for field in ("ticket_type", "priority"):
            with self.subTest(missing=field):
                decision = gate(**{field: None})
                self.assertEqual(decision.route, "escalate")
                self.assertIn(MISSING_PREDICTION, decision.triggers)


class ClassifierFallbackRule(unittest.TestCase):
    def test_fallback_only_noted_when_switch_off(self):
        decision = gate(unavailable_classifiers=("queue",), escalate_on_fallback=False)
        self.assertEqual(decision.route, "auto_draft")
        self.assertIn("Keyword fallback used for: queue", decision.reason)

    def test_fallback_escalates_when_switch_on(self):
        decision = gate(unavailable_classifiers=("queue",), escalate_on_fallback=True)
        self.assertEqual(decision.route, "escalate")
        self.assertEqual(decision.triggers, (CLASSIFIER_UNAVAILABLE,))
        self.assertIn("queue", decision.reason)

    def test_switch_on_with_all_models_present_has_no_effect(self):
        self.assertEqual(gate(escalate_on_fallback=True).route, "auto_draft")


class GraphIntegration(unittest.TestCase):
    """Runs the compiled LangGraph with stubbed classifiers (no model loading)."""

    @classmethod
    def setUpClass(cls):
        try:
            import graph  # noqa: F401
        except ImportError as error:  # langgraph / torch not installed
            raise unittest.SkipTest(f"graph dependencies unavailable: {error}")
        cls.graph_module = graph

    def run_graph(self, type_=("Request", 0.95), queue=("Billing and Payments", 0.95), priority=("low", 0.95)):
        g = self.graph_module
        stubs = {
            "predict_type": lambda *a, **k: type_,
            "predict_queue": lambda *a, **k: queue,
            "predict_priority": lambda *a, **k: priority,
        }
        with mock.patch.multiple(g, **stubs), mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            return g.build_sentinel_graph().invoke({"subject": "Pricing", "body": "What plans do you offer?"})

    def test_confident_ticket_goes_to_retrieve_and_draft(self):
        result = self.run_graph()
        self.assertEqual(result["escalation_status"], "AUTO_DRAFT")
        self.assertEqual(result["escalation_triggers"], [])
        self.assertTrue(result["escalation_reason"].startswith("Passed safety gate"))
        self.assertTrue(result["draft_response"])

    def test_high_priority_incident_goes_to_escalation_with_gate_reason(self):
        result = self.run_graph(type_=("Incident", 0.97), priority=("high", 0.97))
        self.assertEqual(result["escalation_status"], "ESCALATE")
        self.assertEqual(result["escalation_triggers"], [HIGH_PRIORITY_INCIDENT])
        self.assertIn("High-Priority Incident", result["escalation_reason"])

    def test_low_confidence_escalates(self):
        result = self.run_graph(type_=("Request", 0.40), queue=("Billing and Payments", 0.40), priority=("low", 0.40))
        self.assertEqual(result["escalation_status"], "ESCALATE")
        self.assertEqual(result["escalation_triggers"], [LOW_CONFIDENCE])

    def test_classifier_error_is_reported_in_reason(self):
        def broken(*a, **k):
            raise self.graph_module.ModelUnavailableError("queue checkpoint missing")

        g = self.graph_module
        with mock.patch.multiple(
            g,
            predict_type=lambda *a, **k: ("Request", 0.95),
            predict_queue=broken,
            predict_priority=lambda *a, **k: ("low", 0.95),
        ), mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            result = g.build_sentinel_graph().invoke({"subject": "Pricing", "body": "What plans do you offer?"})
        self.assertIn("queue", result["escalation_reason"])


class PipelineIntegration(unittest.TestCase):
    def test_pipeline_uses_shared_gate(self):
        import pipeline

        outage = pipeline.process_ticket("Platform outage", "Critical outage, all services down.")
        self.assertEqual(outage["escalation_status"], "ESCALATE")
        self.assertIn("High-Priority Incident", outage["escalation_reason"])

        pricing = pipeline.process_ticket("Pricing", "I would like to inquire about pricing plans.")
        self.assertEqual(pricing["escalation_status"], "AUTO_DRAFT")


if __name__ == "__main__":
    unittest.main()
