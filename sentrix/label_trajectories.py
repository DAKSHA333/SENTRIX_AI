"""Attach reviewed event labels to anonymous trajectory rows for supervised experiments.

A trajectory row (one person track at one timestamp) gets the event_type of an
annotation in the same clip when its timestamp falls inside [start - pre, end]
and the annotation's track_id is blank (group event) or equals the row's track_id.
Everything else is labelled "normal". Object-centred events (unattended bag) are
skipped because trajectory rows describe people.
"""
import argparse
import csv
import os
from .benchmark import read_csv

OBJECT_EVENTS = {"potentially_unattended_object", "potentially_hazardous_object_detection"}


def label_rows(rows, annotations, manifest=None, pre_seconds=0.0, background="normal"):
    by_clip = {}
    for a in annotations:
        if a["event_type"] in OBJECT_EVENTS:
            continue
        by_clip.setdefault(a["clip_id"], []).append(a)
    splits = {m["clip_id"]: m.get("split", "") for m in (manifest or [])}
    counts = {}
    for row in rows:
        label = background
        t = float(row["timestamp_seconds"])
        for a in by_clip.get(row["clip_id"], []):
            track = str(a.get("track_id") or "").strip()
            if track and track != str(row.get("track_id", "")).strip():
                continue
            if float(a["event_start_seconds"]) - pre_seconds <= t <= float(a["event_end_seconds"]):
                label = a["event_type"]
                break
        row["label"] = label
        if not row.get("split") and row["clip_id"] in splits:
            row["split"] = splits[row["clip_id"]]
        counts[label] = counts.get(label, 0) + 1
    return rows, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trajectories_csv"); p.add_argument("--annotations", required=True)
    p.add_argument("--manifest", help="Fills empty split values from the frozen manifest")
    p.add_argument("--pre-seconds", type=float, default=0.0)
    p.add_argument("--out", default="data/trajectories_labeled.csv")
    a = p.parse_args()
    rows = read_csv(a.trajectories_csv)
    if not rows:
        raise SystemExit("No trajectory rows in {}".format(a.trajectories_csv))
    rows, counts = label_rows(rows, read_csv(a.annotations), read_csv(a.manifest) if a.manifest else None, a.pre_seconds)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    print("Labelled {} rows -> {}: {}".format(len(rows), a.out, counts))

if __name__ == "__main__": main()
