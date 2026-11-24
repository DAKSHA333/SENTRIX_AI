import json
import os
import tempfile
import unittest
from sentrix.config import Settings, Zone
from sentrix.events import EventEngine
from sentrix.tracking import Track

ZONE = Zone("door", [(100, 0), (200, 0), (200, 200), (100, 200)])


def person(x, y=150, ident=1, conf=.9, w=40, h=100):
    """Person whose feet are at (x, y)."""
    return Track(ident, (x - w//2, y - h, x + w//2, y), "person", conf, 0)


class ZoneHysteresisTests(unittest.TestCase):
    def run_positions(self, xs, context=True, **kw):
        engine = EventEngine(Settings(restricted_zones=[ZONE], **kw), 10, evidence_gate=context, context_rules=context)
        found = []
        for i, x in enumerate(xs):
            found += engine.update([person(x)], i, (200, 300, 3), current_time=i / 10)
        return [a["event_type"] for a in found]

    def test_single_jitter_sample_inside_is_not_an_entry_in_context_mode(self):
        xs = [90, 95, 101, 97, 96, 95]
        self.assertEqual(self.run_positions(xs, context=True, zone_entry_confirm_samples=2), [])
        self.assertEqual(self.run_positions(xs, context=False), ["restricted_zone_entry"])

    def test_sustained_entry_alerts_once(self):
        xs = [80, 90, 105, 115, 125, 135, 145]
        self.assertEqual(self.run_positions(xs, zone_entry_confirm_samples=2), ["restricted_zone_entry"])

    def test_foot_point_anchor_uses_where_person_stands(self):
        # Box centre stays left of the zone-top edge region; feet cross a zone that only covers the floor.
        floor = Zone("floor", [(0, 140), (300, 140), (300, 200), (0, 200)])
        engine = EventEngine(Settings(restricted_zones=[floor], zone_entry_confirm_samples=1), 10)
        engine.update([person(50, y=130)], 0, (200, 300, 3), 0.0)
        alerts = engine.update([person(50, y=160)], 1, (200, 300, 3), 0.1)
        self.assertEqual([a["event_type"] for a in alerts], ["restricted_zone_entry"])
        engine = EventEngine(Settings(restricted_zones=[floor], zone_entry_confirm_samples=1, zone_anchor="center"), 10)
        engine.update([person(50, y=130)], 0, (200, 300, 3), 0.0)
        self.assertEqual(engine.update([person(50, y=160)], 1, (200, 300, 3), 0.1), [])


class EvidenceGateTests(unittest.TestCase):
    def entry_alert(self, tracks_before, tracks_after, frame_quality=None):
        engine = EventEngine(Settings(restricted_zones=[ZONE], zone_entry_confirm_samples=1), 10)
        engine.update(tracks_before, 0, (200, 300, 3), 0.0)
        return engine.update(tracks_after, 1, (200, 300, 3), 0.1, frame_quality=frame_quality)

    def test_heavy_overlap_withholds_with_reason(self):
        alerts = self.entry_alert([person(80, ident=1), person(82, ident=2)], [person(130, ident=1), person(132, ident=2)])
        self.assertTrue(alerts)
        self.assertTrue(all(a["event_type"] == "insufficient_evidence" for a in alerts))
        self.assertIn("HEAVY_OVERLAP_WITH_OTHER_PERSON", alerts[0]["evidence_reasons"])

    def test_low_light_withholds_with_reason(self):
        alerts = self.entry_alert([person(80)], [person(130)], frame_quality={"brightness": 12.0, "sharpness": 50.0})
        self.assertEqual(alerts[0]["event_type"], "insufficient_evidence")
        self.assertIn("LOW_LIGHT", alerts[0]["evidence_reasons"])
        ok = self.entry_alert([person(80)], [person(130)], frame_quality={"brightness": 120.0, "sharpness": 50.0})
        self.assertEqual(ok[0]["event_type"], "restricted_zone_entry")
        self.assertEqual(ok[0]["evidence_reasons"], [])


class UnattendedObjectTests(unittest.TestCase):
    def simulate(self, context, owner_positions, seconds=14, fps=5):
        s = Settings(unattended_seconds=8, association_distance_px=90, separation_seconds=2)
        engine = EventEngine(s, fps, evidence_gate=context, context_rules=context)
        bag = Track(101, (200, 170, 224, 196), "backpack", .8, 0)
        out = []
        for i in range(seconds * fps):
            t = i / fps
            x = owner_positions(t)
            tracks = [bag] + ([person(x)] if x is not None else [])
            out += engine.update(tracks, i, (360, 640, 3), t)
        return out

    def test_owner_staying_nearby_is_not_unattended_in_context_mode(self):
        stays = lambda t: 230
        self.assertEqual(self.simulate(True, stays), [])
        single = self.simulate(False, stays)
        self.assertEqual([a["reason_code"] for a in single], ["STATIONARY_OBJECT_DURATION"])

    def test_owner_leaving_frame_triggers_after_both_criteria(self):
        leaves = lambda t: 230 + 80 * t if t < 3 else None
        alerts = self.simulate(True, leaves)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["event_type"], "potentially_unattended_object")
        self.assertGreaterEqual(alerts[0]["measurements"]["stationary_seconds"], 8)

    def test_borderline_person_makes_association_ambiguous(self):
        hovering = lambda t: 230 if t < 2 else 290   # centre ~114 px from bag: beyond 90, within 1.5x
        alerts = self.simulate(True, hovering)
        self.assertEqual(alerts[0]["event_type"], "insufficient_evidence")
        self.assertIn("PERSON_ASSOCIATION_AMBIGUOUS", alerts[0]["evidence_reasons"])


class ExtensionRobustnessTests(unittest.TestCase):
    def test_jitter_on_standing_person_is_not_pacing(self):
        import random
        rng = random.Random(3)
        engine = EventEngine(Settings(enable_dwell_pacing=True, pacing_reversals=3), 10)
        found = []
        for i in range(200):
            x = 300 + rng.gauss(0, 4)
            found += engine.update([person(int(x))], i, (360, 640, 3), i / 10)
        self.assertEqual(found, [])

    def test_stale_track_state_is_pruned(self):
        engine = EventEngine(Settings(restricted_zones=[ZONE], state_ttl_seconds=1), 10)
        engine.update([person(50, ident=7)], 0, (200, 300, 3), 0.0)
        for i in range(1, 60):
            engine.update([], i, (200, 300, 3), i / 10)
        self.assertNotIn(7, engine.history)
        self.assertFalse(any(k[0] == 7 for k in engine.previous_zone_state))


class ConfigTests(unittest.TestCase):
    def test_unknown_key_is_rejected_and_comments_are_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "c.json")
            with open(path, "w") as f: json.dump({"_comment": "fine", "unattended_seconds": 5}, f)
            self.assertEqual(Settings.from_json(path).unattended_seconds, 5)
            with open(path, "w") as f: json.dump({"unatended_seconds": 5}, f)
            with self.assertRaises(ValueError):
                Settings.from_json(path)


if __name__ == "__main__": unittest.main()
