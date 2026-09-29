import unittest

from src.cv_engine.state_builder import ObservationValidator, StateBuilder
from src.protocol_engine.engine import FirstAidProtocolEngine


def obs(t, blood=8.0, hand=False, cloth=False, conf=0.9, wound=True):
    return {
        "timestamp": float(t), "wound_visible": wound, "blood_detected": blood > 1.5,
        "bleeding_severity": "severe" if blood > 6 else ("moderate" if blood > 1.5 else "none"),
        "blood_pixels_pct": blood, "hand_near_wound": hand, "cloth_on_wound": cloth,
        "detection_confidence": conf, "detector_id": "test", "threshold_set": "test",
    }


class FakeTracker:
    def get_hand_landmarks(self, frame):
        return []
    def close(self):
        pass


class TestValidator(unittest.TestCase):
    def test_bad_confidence_discarded(self):
        ok, reason = ObservationValidator().validate(obs(0, conf=0.2))
        self.assertFalse(ok)
        self.assertEqual(reason, "confidence")

    def test_contradiction_discarded(self):
        x = obs(0, blood=0)
        x["bleeding_severity"] = "severe"
        x["blood_detected"] = False
        ok, reason = ObservationValidator().validate(x)
        self.assertFalse(ok)
        self.assertEqual(reason, "consistency")


class TestStateBuilder(unittest.TestCase):
    def test_sustained_pressure_and_reversal(self):
        b = StateBuilder(tracker=FakeTracker(), expected_hz=2)
        for t in [0, 0.5, 1, 1.5, 2, 2.5]:
            facts = b.add_observation(obs(t, blood=8, hand=True))
        self.assertTrue(facts["pressure_applied"])
        facts = b.add_observation(obs(3, blood=8, hand=False))
        self.assertTrue(facts["pressure_recently_removed"])

    def test_time_window_prunes_old_evidence(self):
        b = StateBuilder(tracker=FakeTracker(), buffer_seconds=5, expected_hz=2)
        b.add_observation(obs(0, blood=8))
        b.add_observation(obs(6, blood=0))
        self.assertEqual(len(b.observations), 1)


class TestRules(unittest.TestCase):
    def facts(self, **overrides):
        f = {
            "wound_visible": True, "confidence": 1.0, "bleeding": "severe",
            "pressure_applied": False, "pressure_recently_removed": False,
            "gauze_present": False, "time_since_last_change_s": 0,
        }
        f.update(overrides)
        return f

    def test_key_reversal_rule(self):
        e = FirstAidProtocolEngine()
        d = e.evaluate(self.facts(pressure_applied=True), 1)
        self.assertEqual(d["rule_fired"], "R-85")
        d = e.evaluate(self.facts(pressure_applied=False, pressure_recently_removed=True), 2)
        self.assertEqual(d["rule_fired"], "R-97")

    def test_controlled_without_gauze(self):
        e = FirstAidProtocolEngine()
        d = e.evaluate(self.facts(bleeding="controlled"), 1)
        self.assertEqual(d["rule_fired"], "R-75")

    def test_controlled_with_gauze(self):
        e = FirstAidProtocolEngine()
        d = e.evaluate(self.facts(bleeding="controlled", gauze_present=True), 1)
        self.assertEqual(d["rule_fired"], "R-70")

    def test_low_confidence_precedes_specific_rule(self):
        e = FirstAidProtocolEngine()
        d = e.evaluate(self.facts(confidence=0.2), 1)
        self.assertEqual(d["rule_fired"], "R-99")

    def test_no_wound_precedes_low_confidence(self):
        e = FirstAidProtocolEngine()
        d = e.evaluate(self.facts(wound_visible=False, confidence=0.2), 1)
        self.assertEqual(d["rule_fired"], "R-100")


if __name__ == "__main__":
    unittest.main()

class TestEndToEnd(unittest.TestCase):
    def test_adaptive_sequence(self):
        b = StateBuilder(tracker=FakeTracker(), expected_hz=2)
        e = FirstAidProtocolEngine()
        seen = []
        for i, hand in enumerate([False, False, True, True, True, True, True, False]):
            t = i * 0.5
            facts = b.add_observation(obs(t, blood=8, hand=hand))
            seen.append(e.evaluate(facts, t)["rule_fired"])
        self.assertIn("R-85", seen)
        self.assertIn("R-97", seen)
