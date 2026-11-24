import unittest
import numpy as np
from sentrix.analyze import SceneMonitor
from sentrix.config import Settings
from sentrix.events import EventEngine
from sentrix.tracking import Track


def person(ident, x, y=300, h=200):
    return Track(ident, (x - 40, y - h, x + 40, y), "person", .9, 0)


def run(engine, frames, fps=10):
    out = []
    for i, kw in enumerate(frames):
        out += engine.update(kw.get("tracks", []), i, (480, 640, 3), i / fps, frame_quality=kw.get("fq"), pose=kw.get("pose"))
    return [a["event_type"] for a in out], out


class ObjectAlertTests(unittest.TestCase):
    def test_confirmation_and_cooldown(self):
        s = Settings(object_confirm_samples=3, object_alert_cooldown_seconds=10, low_quality_min_object_height=5)
        knife = lambda i: Track(i, (100, 100, 160, 140), "knife", .7, 0)
        types, _ = run(EventEngine(s, 10), [{"tracks": [knife(1)]}] * 2)
        self.assertEqual(types, [])
        frames = [{"tracks": [knife(1)]}] * 3 + [{"tracks": [knife(2)]}] * 5      # new ID inside cooldown
        types, alerts = run(EventEngine(s, 10), frames)
        self.assertEqual(types, ["potentially_hazardous_object_detection"])
        self.assertEqual(alerts[0]["measurements"]["detections_before_alert"], 3)

    def test_custom_weapon_label_is_supported(self):
        s = Settings(alert_object_classes=("knife", "pistol"), low_quality_min_object_height=5)
        types, alerts = run(EventEngine(s, 10), [{"tracks": [Track(1_000_001, (10, 10, 90, 60), "pistol", .8, 0)]}])
        self.assertEqual(alerts[0]["measurements"]["object_label"], "pistol")


class TamperTests(unittest.TestCase):
    def test_covered_lens_alerts_once_after_sustain_and_rearms(self):
        s = Settings(enable_tamper=True, tamper_sustain_seconds=1.0)
        ok, dark = {"brightness": 110, "sharpness": 80, "changed_fraction": 0.1}, {"brightness": 4, "sharpness": 1, "changed_fraction": 0.9}
        frames = [{"fq": ok}] * 5 + [{"fq": dark}] * 15 + [{"fq": ok}] * 15 + [{"fq": dark}] * 15
        types, alerts = run(EventEngine(s, 10), frames)
        self.assertEqual(types, ["camera_tampering_suspected", "camera_tampering_suspected"])
        self.assertIn("VIEW_DARK", alerts[0]["measurements"]["cues"])

    def test_close_up_person_alone_is_not_tampering(self):
        s = Settings(enable_tamper=True, tamper_sustain_seconds=1.0)
        close = {"brightness": 100, "sharpness": 60, "changed_fraction": 0.9}
        self.assertEqual(run(EventEngine(s, 10), [{"fq": close}] * 30)[0], [])
        s.tamper_detect_moved = True
        self.assertEqual(run(EventEngine(s, 10), [{"fq": close}] * 30)[0], ["camera_tampering_suspected"])

    def test_scene_monitor_separates_partial_and_full_view_change(self):
        mon = SceneMonitor()
        base = np.full((360, 640, 3), 120, np.uint8)
        mon.update(base)
        person_frame = base.copy(); person_frame[:, :200] = 30
        self.assertLess(mon.update(person_frame)["changed_fraction"], 0.5)
        covered = np.full((360, 640, 3), 5, np.uint8)
        q = mon.update(covered)
        self.assertGreater(q["changed_fraction"], 0.9)
        self.assertLess(q["brightness"], 12)


class CrowdRunningPoseTests(unittest.TestCase):
    def test_crowding(self):
        s = Settings(enable_crowd=True, crowd_alert_count=3, crowd_sustain_seconds=1.0)
        crowd = [person(i, 100 + 120 * i) for i in range(3)]
        types, _ = run(EventEngine(s, 10), [{"tracks": crowd}] * 15)
        self.assertEqual(types, ["crowding_threshold_exceeded"])
        types, _ = run(EventEngine(s, 10), [{"tracks": crowd[:2]}] * 15)
        self.assertEqual(types, [])

    def test_running_uses_body_heights_per_second(self):
        s = Settings(enable_running=True, running_body_heights_per_s=1.6, running_sustain_seconds=0.5)
        fast = [{"tracks": [person(1, 50 + 40 * i)]} for i in range(12)]    # 400 px/s on a 200 px person = 2 bh/s
        slow = [{"tracks": [person(1, 50 + 10 * i)]} for i in range(12)]    # 0.5 bh/s
        self.assertEqual(run(EventEngine(s, 10), fast)[0], ["rapid_movement"])
        self.assertEqual(run(EventEngine(s, 10), slow)[0], [])

    def test_both_arms_raised_needs_sustained_pose(self):
        s = Settings(enable_arms_raised=True, arms_raised_sustain_seconds=1.0)
        up = {"tracks": [person(1, 300)], "pose": {1: {"arms_raised": 2, "visible_keypoints": 15}}}
        one = {"tracks": [person(1, 300)], "pose": {1: {"arms_raised": 1, "visible_keypoints": 15}}}
        self.assertEqual(run(EventEngine(s, 10), [up] * 15)[0], ["both_arms_raised"])
        self.assertEqual(run(EventEngine(s, 10), [one] * 15)[0], [])
        self.assertEqual(run(EventEngine(s, 10), [up, one] * 8)[0], [])


if __name__ == "__main__": unittest.main()
