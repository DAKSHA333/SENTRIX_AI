"""Evaluate event alerts against scene/session-separated annotations."""
import argparse
import csv
import json
import math
import os
import statistics

BREAKDOWN_FIELDS = ("illumination", "crowd_density", "viewpoint")
NOT_ALERTS = ("insufficient_evidence",)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def wilson(successes, total, z=1.96):
    if not total:
        return [None, None]
    p = successes / total
    den = 1 + z*z/total
    mid = (p + z*z/(2*total))/den
    half = z*math.sqrt((p*(1-p)+z*z/(4*total))/total)/den
    return [round(max(0, mid-half), 4), round(min(1, mid+half), 4)]


def _truthy(value):
    return str(value or "").strip().lower() in ("true", "1", "yes")


def _prf(tp, fp, fn):
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return precision, recall, f1


def _can_match(prediction, truth):
    """Typed alerts match only their own class; motion alarms are untyped candidates."""
    if prediction["clip_id"] != truth["clip_id"]:
        return False
    if prediction["event_type"] == "motion_alarm":
        return str(prediction.get("method", "motion")) in ("", "motion")
    return prediction["event_type"] == truth["event_type"]


def match_alerts(annotations, predictions, tolerance=2.0):
    """Greedy one-to-one matching in time order, within each clip.

    Returns (matched: {truth_index: (prediction, delay)}, unmatched_predictions).
    """
    candidates = sorted((p for p in predictions if p["event_type"] not in NOT_ALERTS),
                        key=lambda r: (r["clip_id"], float(r["timestamp_seconds"])))
    matched, unmatched = {}, []
    for p in candidates:
        t = float(p["timestamp_seconds"])
        options = []
        for gi, g in enumerate(annotations):
            if gi in matched or not _can_match(p, g):
                continue
            start, end = float(g["event_start_seconds"]), float(g["event_end_seconds"])
            if start - tolerance <= t <= end + tolerance:
                trigger = float(g.get("trigger_seconds") or start)
                options.append((abs(t - trigger), gi, max(0.0, t - trigger)))
        if options:
            _, gi, delay = min(options)
            matched[gi] = (p, delay)
        else:
            unmatched.append(p)
    return matched, unmatched


CORE_EVENTS = ("restricted_zone_entry", "potentially_unattended_object")


def evaluate(annotations, predictions, manifest, tolerance=2.0, throughput_fps=None, split="test", event_types=None):
    """Score one method's predictions.

    split: "test" (default, held-out reporting), "development" (tuning), or "all".
    event_types: optional collection restricting the evaluation to these classes
    (motion alarms and abstentions for those classes are kept).
    """
    wanted = (lambda s: True) if split == "all" else (lambda s: s == split)
    clips = {r["clip_id"]: r for r in manifest if wanted(str(r.get("split") or "test").lower())}
    ann = [r for r in annotations if r["clip_id"] in clips]
    pred = [r for r in predictions if r["clip_id"] in clips]
    if event_types:
        keep = set(event_types)
        ann = [r for r in ann if r["event_type"] in keep]
        pred = [r for r in pred if r["event_type"] in keep or r["event_type"] == "motion_alarm"
                or (r["event_type"] == "insufficient_evidence" and r.get("candidate_event_type") in keep)]
    candidates = [p for p in pred if p["event_type"] not in NOT_ALERTS]
    matched, unmatched = match_alerts(ann, pred, tolerance)

    by_event = {}
    for event in sorted({r["event_type"] for r in ann} | {p["event_type"] for p in unmatched if p["event_type"] != "motion_alarm"}):
        gt_idx = [i for i, g in enumerate(ann) if g["event_type"] == event]
        tp = sum(1 for i in gt_idx if i in matched)
        fn = len(gt_idx) - tp
        # Typed unmatched alerts count against their class. Untyped motion alarms remain
        # part of the overall false-alerts/hour rate rather than inventing a class.
        fp = sum(1 for p in unmatched if p["event_type"] == event)
        precision, recall, f1 = _prf(tp, fp, fn)
        delays = [matched[i][1] for i in gt_idx if i in matched]
        by_event[event] = {"ground_truth_events": len(gt_idx), "matched_alerts": tp, "false_alerts": fp, "misses": fn,
            "precision": round(precision, 4), "precision_wilson_95": wilson(tp, tp+fp),
            "recall": round(recall, 4), "recall_wilson_95": wilson(tp, tp+fn), "f1": round(f1, 4),
            "median_alert_delay_seconds": round(statistics.median(delays), 3) if delays else None}

    # Abstention quality is measured against intervals explicitly marked as low visibility.
    lowvis = [g for g in ann if _truthy(g.get("low_visibility"))]
    abstentions = [p for p in pred if p["event_type"] == "insufficient_evidence"]
    abstention_hits = 0
    for g in lowvis:
        s, e = float(g["event_start_seconds"]), float(g["event_end_seconds"])
        if any(p["clip_id"] == g["clip_id"] and s-tolerance <= float(p["timestamp_seconds"]) <= e+tolerance for p in abstentions):
            abstention_hits += 1
    # Abstentions raised where nothing was annotated are wasted reviewer attention, reported separately.
    spurious_abstentions = 0
    for p in abstentions:
        t = float(p["timestamp_seconds"])
        if not any(g["clip_id"] == p["clip_id"] and float(g["event_start_seconds"])-tolerance <= t <= float(g["event_end_seconds"])+tolerance for g in ann):
            spurious_abstentions += 1

    hours = sum(float(r.get("duration_seconds") or 0) for r in clips.values())/3600.0
    delays = [d for _, d in matched.values()]
    # Macro-F1 averages annotated classes only; alert classes with no ground truth still add false alerts.
    typed = sorted({g["event_type"] for g in ann})
    overall_tp = len(matched); overall_fn = len(ann)-overall_tp; overall_fp = len(unmatched)
    overall_precision, overall_recall, overall_f1 = _prf(overall_tp, overall_fp, overall_fn)
    generic_motion = bool(candidates) and all(p["event_type"] == "motion_alarm" for p in candidates)
    ratings = [float(p["explanation_rating"]) for p in pred if p.get("explanation_rating") not in (None, "", "None")]
    result = {"split": split, "event_types": sorted(event_types) if event_types else "all", "clips": len(clips), "annotation_events": len(ann), "eligible_alerts": len(candidates),
        "false_alerts": overall_fp,
        "false_alerts_per_camera_hour": round(overall_fp/hours, 4) if hours else None,
        "evaluated_camera_hours": round(hours, 4),
        "macro_f1": None if generic_motion else (round(statistics.mean(by_event[e]["f1"] for e in typed), 4) if typed else None),
        "event_detection_precision": round(overall_precision, 4), "event_detection_recall": round(overall_recall, 4),
        "event_detection_f1": round(overall_f1, 4),
        "median_alert_delay_seconds": round(statistics.median(delays), 3) if delays else None,
        "p95_alert_delay_seconds": round(sorted(delays)[min(len(delays)-1, math.ceil(.95*len(delays))-1)], 3) if delays else None,
        "low_visibility_intervals": len(lowvis),
        "appropriate_abstention_rate": round(abstention_hits/len(lowvis), 4) if lowvis else None,
        "abstentions": len(abstentions), "abstentions_outside_annotated_events": spurious_abstentions,
        "explanation_usefulness_mean": round(statistics.mean(ratings), 3) if ratings else None,
        "explanation_ratings_count": len(ratings),
        "throughput_fps": throughput_fps, "events": by_event,
        "breakdown": breakdown(clips, ann, matched, unmatched),
        "limitations": "Confidence intervals are per-event Wilson intervals, not scene-clustered intervals. Use sentrix.compare for a clip-level paired bootstrap. Independent scenes/sessions are required for generalization claims."}
    return result


def parse_events(text):
    if not text:
        return None
    return CORE_EVENTS if text.strip().lower() == "core" else tuple(x.strip() for x in text.split(",") if x.strip())


def breakdown(clips, ann, matched, unmatched):
    """Robustness breakdown by recorded viewing conditions (proposal section 13)."""
    out = {}
    for field in BREAKDOWN_FIELDS:
        values = sorted({(c.get(field) or "unspecified") for c in clips.values()})
        if values == ["unspecified"]:
            continue
        table = {}
        for value in values:
            ids = {cid for cid, c in clips.items() if (c.get(field) or "unspecified") == value}
            gt = [i for i, g in enumerate(ann) if g["clip_id"] in ids]
            tp = sum(1 for i in gt if i in matched)
            fp = sum(1 for p in unmatched if p["clip_id"] in ids)
            hours = sum(float(clips[c].get("duration_seconds") or 0) for c in ids)/3600.0
            table[value] = {"clips": len(ids), "ground_truth_events": len(gt), "matched": tp,
                            "recall": round(tp/len(gt), 4) if gt else None, "false_alerts": fp,
                            "false_alerts_per_camera_hour": round(fp/hours, 4) if hours else None}
        out[field] = table
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="reports/evaluation.json")
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--throughput-fps", type=float)
    parser.add_argument("--split", default="test", choices=("test", "development", "train", "all"))
    parser.add_argument("--events", help="Comma-separated event types to evaluate, or 'core' for the two core events")
    args = parser.parse_args()
    report = evaluate(read_csv(args.annotations), read_csv(args.predictions), read_csv(args.manifest), args.tolerance,
                      args.throughput_fps, args.split, parse_events(args.events))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle: json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
