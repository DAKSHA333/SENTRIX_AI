import copy
import os
import tempfile
import unittest
from sentrix.benchmark import evaluate, read_csv
from sentrix.compare import paired_bootstrap
from sentrix.label_trajectories import label_rows


def ann(clip, event="restricted_zone_entry", start=10, end=12, low="false"):
    return {"clip_id": clip, "event_type": event, "event_start_seconds": str(start), "event_end_seconds": str(end),
            "trigger_seconds": str(start), "low_visibility": low}


def pred(clip, t, event="restricted_zone_entry", candidate=""):
    return {"clip_id": clip, "event_type": event, "timestamp_seconds": str(t), "candidate_event_type": candidate, "method": "context"}


MANIFEST = [{"clip_id": c, "split": "test", "duration_seconds": "60"} for c in ("a", "b")]


class BenchmarkFixTests(unittest.TestCase):
    def test_alert_in_one_clip_never_matches_ground_truth_in_another(self):
        result = evaluate([ann("a")], [pred("b", 11)], MANIFEST)
        self.assertEqual(result["events"]["restricted_zone_entry"]["matched_alerts"], 0)
        self.assertEqual(result["false_alerts"], 1)

    def test_low_visibility_abstention_is_scored_without_crashing(self):
        result = evaluate([ann("a", low="true")], [pred("a", 10.5, "insufficient_evidence", "restricted_zone_entry")], MANIFEST)
        self.assertEqual(result["appropriate_abstention_rate"], 1.0)
        self.assertEqual(result["events"]["restricted_zone_entry"]["misses"], 1)

    def test_event_filter_and_breakdown(self):
        manifest = [dict(m, illumination=("dim" if m["clip_id"] == "a" else "normal")) for m in MANIFEST]
        result = evaluate([ann("a"), ann("b", "sudden_group_dispersion")], [pred("a", 10), pred("b", 10, "sudden_group_dispersion")],
                          manifest, event_types=("restricted_zone_entry",))
        self.assertEqual(result["annotation_events"], 1)
        self.assertEqual(list(result["events"]), ["restricted_zone_entry"])
        self.assertEqual(result["breakdown"]["illumination"]["dim"]["recall"], 1.0)

    def test_split_selection(self):
        manifest = [{"clip_id": "a", "split": "train", "duration_seconds": "60"}, {"clip_id": "b", "split": "test", "duration_seconds": "60"}]
        self.assertEqual(evaluate([ann("a"), ann("b")], [], manifest)["annotation_events"], 1)
        self.assertEqual(evaluate([ann("a"), ann("b")], [], manifest, split="all")["annotation_events"], 2)

    def test_paired_bootstrap_reports_difference_intervals(self):
        manifest = [{"clip_id": c, "split": "test", "duration_seconds": "60"} for c in "abcd"]
        annotations = [ann(c) for c in "abcd"]
        good = [pred(c, 10.5) for c in "abcd"]
        noisy = good + [pred(c, 40) for c in "abcd"]
        boot = paired_bootstrap(annotations, manifest, {"single_rule": noisy, "context": good}, "single_rule", iterations=50)
        lo, hi = boot["differences_vs_reference"]["context"]["false_alerts_per_hour_delta_95ci"]
        self.assertLess(hi, 0)


class TrajectoryLabelTests(unittest.TestCase):
    def test_rows_get_event_or_normal_labels(self):
        rows = [{"clip_id": "a", "track_id": "9", "timestamp_seconds": "5", "split": ""},
                {"clip_id": "a", "track_id": "1", "timestamp_seconds": "5", "split": ""},
                {"clip_id": "a", "track_id": "9", "timestamp_seconds": "20", "split": ""}]
        annotations = [dict(ann("a", "sustained_counter_flow", 3, 10), track_id="9"),
                       dict(ann("a", "potentially_unattended_object", 0, 30), track_id="101")]
        labelled, counts = label_rows(rows, annotations, [{"clip_id": "a", "split": "train"}])
        self.assertEqual([r["label"] for r in labelled], ["sustained_counter_flow", "normal", "normal"])
        self.assertEqual(labelled[0]["split"], "train")


class EndToEndSyntheticTests(unittest.TestCase):
    def test_simulated_abandoned_bag_flows_through_analyze_store_and_evidence(self):
        from sentrix.analyze import analyze
        from sentrix.config import Settings
        from sentrix.export import export_db
        from sentrix.simulate import generate
        with tempfile.TemporaryDirectory() as root:
            data = os.path.join(root, "syn")
            manifest, annotations = generate(data, seeds=1, scenarios=("bag_abandon", "bag_benign_stay"))
            settings = Settings.from_json(os.path.join(data, "config.synthetic.json"))
            settings.evidence_dir = os.path.join(root, "evidence"); settings.trajectory_csv = os.path.join(root, "traj.csv")
            db = os.path.join(root, "context.db")
            for m in manifest:
                analyze(m["video_path"], None, method="context", clip_id=m["clip_id"], database_path=db,
                        detections=m["video_path"].replace(".mp4", ".tracks.json"), settings=copy.deepcopy(settings), quiet=True)
            out = os.path.join(root, "pred.csv")
            export_db(db, out)
            preds = read_csv(out)
            unattended = [p for p in preds if p["event_type"] == "potentially_unattended_object"]
            self.assertEqual([p["clip_id"] for p in unattended], ["syn_bag_abandon_0"])
            result = evaluate(annotations, preds, manifest, split="all")
            self.assertEqual(result["events"]["potentially_unattended_object"]["matched_alerts"], 1)
            import sqlite3
            paths = [r[0] for r in sqlite3.connect(db).execute("SELECT evidence_path FROM alerts")]
            self.assertTrue(paths and all(p and os.path.isfile(p) for p in paths))


if __name__ == "__main__": unittest.main()
