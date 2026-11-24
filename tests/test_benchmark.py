import unittest
from sentrix.benchmark import evaluate


class BenchmarkTests(unittest.TestCase):
    def test_alert_matching_and_camera_hour_denominator(self):
        annotations=[{"clip_id":"c1","event_type":"restricted_zone_entry","event_start_seconds":"10","event_end_seconds":"12","trigger_seconds":"10","low_visibility":"false"}]
        predictions=[{"clip_id":"c1","event_type":"restricted_zone_entry","timestamp_seconds":"11","evidence_quality":"SUFFICIENT","candidate_event_type":"","method":"context"},
                     {"clip_id":"c1","event_type":"restricted_zone_entry","timestamp_seconds":"40","evidence_quality":"SUFFICIENT","candidate_event_type":"","method":"context"}]
        manifest=[{"clip_id":"c1","split":"test","duration_seconds":"3600"}]
        result=evaluate(annotations,predictions,manifest)
        row=result["events"]["restricted_zone_entry"]
        self.assertEqual(row["matched_alerts"],1)
        self.assertEqual(row["false_alerts"],1)
        self.assertEqual(result["false_alerts_per_camera_hour"],1.0)

    def test_motion_alarm_matches_any_temporal_event_once(self):
        annotations=[{"clip_id":"c1","event_type":"restricted_zone_entry","event_start_seconds":"10","event_end_seconds":"11","trigger_seconds":"10","low_visibility":"false"}]
        predictions=[{"clip_id":"c1","event_type":"motion_alarm","timestamp_seconds":"10.5","method":"motion"}]
        result=evaluate(annotations,predictions,[{"clip_id":"c1","split":"test","duration_seconds":"60"}])
        self.assertEqual(result["events"]["restricted_zone_entry"]["matched_alerts"],1)
        self.assertEqual(result["false_alerts_per_camera_hour"],0.0)
        self.assertIsNone(result["macro_f1"])

    def test_motion_false_alerts_reduce_overall_event_precision(self):
        annotations=[{"clip_id":"c1","event_type":"restricted_zone_entry","event_start_seconds":"10","event_end_seconds":"11","trigger_seconds":"10","low_visibility":"false"}]
        predictions=[{"clip_id":"c1","event_type":"motion_alarm","timestamp_seconds":"10.5","method":"motion"},
                     {"clip_id":"c1","event_type":"motion_alarm","timestamp_seconds":"40","method":"motion"}]
        result=evaluate(annotations,predictions,[{"clip_id":"c1","split":"test","duration_seconds":"3600"}])
        self.assertEqual(result["event_detection_precision"],0.5)
        self.assertEqual(result["event_detection_f1"],0.6667)


if __name__=="__main__":unittest.main()
