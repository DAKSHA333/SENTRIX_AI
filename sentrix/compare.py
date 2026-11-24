"""Compare exported prediction CSVs from frozen baseline runs on one test set.

Adds a clip-level paired bootstrap so the H1 comparison (context vs. single rule vs.
motion) reports an interval for the *difference* in F1 and false alerts/hour, not
just two point estimates.
"""
import argparse
import json
import os
import random
from .benchmark import evaluate, read_csv, parse_events


def _resample(rows_by_clip, picks):
    out = []
    for k, cid in enumerate(picks):
        for row in rows_by_clip.get(cid, []):
            copy = dict(row); copy["clip_id"] = "{}#{}".format(cid, k); out.append(copy)
    return out


def paired_bootstrap(annotations, manifest, runs, reference, split="test", iterations=1000, seed=7, tolerance=2.0, event_types=None):
    wanted = (lambda s: True) if split == "all" else (lambda s: s == split)
    clips = [r for r in manifest if wanted(str(r.get("split") or "test").lower())]
    if len(clips) < 2 or reference not in runs:
        return None
    ids = [r["clip_id"] for r in clips]
    man = {r["clip_id"]: [r] for r in clips}
    ann = {}
    for r in annotations: ann.setdefault(r["clip_id"], []).append(r)
    preds = {}
    for name, rows in runs.items():
        preds[name] = {}
        for r in rows: preds[name].setdefault(r["clip_id"], []).append(r)
    rng = random.Random(seed)
    deltas = {name: {"f1": [], "fa": []} for name in runs if name != reference}
    for _ in range(iterations):
        picks = [rng.choice(ids) for _ in ids]
        m = _resample(man, picks)
        for row in m: row["split"] = "test"
        a = _resample(ann, picks)
        ref = evaluate(a, _resample(preds[reference], picks), m, tolerance, event_types=event_types)
        for name in deltas:
            res = evaluate(a, _resample(preds[name], picks), m, tolerance, event_types=event_types)
            deltas[name]["f1"].append(res["event_detection_f1"] - ref["event_detection_f1"])
            if res["false_alerts_per_camera_hour"] is not None and ref["false_alerts_per_camera_hour"] is not None:
                deltas[name]["fa"].append(res["false_alerts_per_camera_hour"] - ref["false_alerts_per_camera_hour"])
    def ci(values):
        if not values: return None
        s = sorted(values)
        return [round(s[int(.025*(len(s)-1))], 4), round(s[int(.975*(len(s)-1))], 4)]
    return {"reference": reference, "iterations": iterations, "resampling_unit": "clip", "clips": len(ids),
            "differences_vs_reference": {name: {"event_f1_delta_95ci": ci(d["f1"]),
                                                "false_alerts_per_hour_delta_95ci": ci(d["fa"])} for name, d in deltas.items()},
            "note": "Positive F1 delta and negative false-alert delta favour the method over the reference. Clip resampling ignores scene clustering; with few scenes, intervals are optimistic."}


def compare_runs(annotations, manifest, runs, out, reference="single_rule", split="test", bootstrap=1000,
                 tolerance=2.0, event_types=None, quiet=False):
    result = {name: evaluate(annotations, rows, manifest, tolerance, split=split, event_types=event_types) for name, rows in runs.items()}
    if bootstrap > 0:
        boot = paired_bootstrap(annotations, manifest, runs, reference, split, bootstrap, tolerance=tolerance, event_types=event_types)
        if boot: result["_paired_bootstrap"] = boot
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f: json.dump(result, f, indent=2)
    names = [n for n in result if not n.startswith("_")]
    if not quiet:
        print(format_table(result))
        if "_paired_bootstrap" in result:
            print("Paired bootstrap vs {}: {}".format(reference, json.dumps(result["_paired_bootstrap"]["differences_vs_reference"])))
    try:
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(out) or ".", ".mplconfig"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f1 = [result[n]["event_detection_f1"] or 0 for n in names]; fa = [result[n]["false_alerts_per_camera_hour"] or 0 for n in names]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        label = "core events" if event_types and set(event_types) == {"restricted_zone_entry", "potentially_unattended_object"} else "events"
        axes[0].bar(names, f1, color="#315a7d"); axes[0].set_title("Event-detection F1 ({}, {} split)".format(label, split), fontsize=10); axes[0].set_ylim(0, 1.08)
        axes[1].bar(names, fa, color="#c87344"); axes[1].set_title("False alerts / camera-hour", fontsize=10)
        for ax, vals in zip(axes, (f1, fa)):
            ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
            for i, v in enumerate(vals): ax.annotate("{:.2f}".format(v) if v < 100 else "{:.0f}".format(v), (i, v), ha="center", va="bottom", fontsize=9)
        fig.tight_layout(); chart = os.path.splitext(out)[0]+".png"; fig.savefig(chart, dpi=160); plt.close(fig)
        result["_chart"] = chart
        if not quiet: print("Saved comparison chart to {}".format(chart))
    except ImportError:
        pass
    return result


def format_table(result):
    names = [n for n in result if not n.startswith("_")]
    lines = ["{:<14}{:>8}{:>8}{:>8}{:>10}{:>14}{:>11}".format("method", "prec", "recall", "F1", "macroF1", "FA/cam-hour", "withheld")]
    for n in names:
        r = result[n]
        lines.append("{:<14}{:>8}{:>8}{:>8}{:>10}{:>14}{:>11}".format(n, r["event_detection_precision"], r["event_detection_recall"],
              r["event_detection_f1"], r["macro_f1"] if r["macro_f1"] is not None else "n/a",
              r["false_alerts_per_camera_hour"] if r["false_alerts_per_camera_hour"] is not None else "n/a", r["abstentions"]))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--run", action="append", required=True, help="NAME=predictions.csv; pass once per method")
    p.add_argument("--reference", default="single_rule", help="Method the others are compared against (bootstrap)")
    p.add_argument("--split", default="test", choices=("test", "development", "train", "all"))
    p.add_argument("--bootstrap", type=int, default=1000, help="Paired bootstrap iterations; 0 disables")
    p.add_argument("--tolerance", type=float, default=2.0)
    p.add_argument("--events", help="Comma-separated event types, or 'core' (restricted_zone_entry, potentially_unattended_object)")
    p.add_argument("--out", default="reports/method_comparison.json")
    args = p.parse_args()
    runs = {}
    for item in args.run:
        name, path = item.split("=", 1); runs[name] = read_csv(path)
    compare_runs(read_csv(args.annotations), read_csv(args.manifest), runs, args.out, args.reference, args.split,
                 args.bootstrap, args.tolerance, parse_events(args.events))

if __name__ == "__main__": main()
