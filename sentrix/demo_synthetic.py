"""One-command end-to-end check on SYNTHETIC scenes (no model weights, no footage needed).

    python -m sentrix.demo_synthetic

1. generates scripted scenes + noisy simulated tracks + ground truth (sentrix.simulate)
2. runs motion / single_rule / context through the real analyze() pipeline
3. exports alerts, scores them (core events and all events), paired bootstrap
4. writes face-blurred annotated showcase videos for a few clips
5. labels trajectories and trains the scikit-learn baseline classifier
6. writes reports/synthetic/SUMMARY.md

Everything here verifies software behaviour. None of it is a research result.
"""
import argparse
import copy
import json
import os
import shutil
import time
from .analyze import analyze, METHODS
from .benchmark import read_csv, CORE_EVENTS
from .compare import compare_runs, format_table
from .config import Settings
from .export import export_db
from .simulate import generate

SHOWCASE = ("bag_abandon", "bag_benign_stay", "boundary_linger", "entry_lowvis", "counterflow")
BANNER = "SYNTHETIC VERIFICATION ONLY - scripted rectangles and simulated detections; not evidence of real-world performance."


def run(data_dir="data/synthetic", report_dir="reports/synthetic", seeds=5, regenerate=False, bootstrap=500):
    t0 = time.monotonic()
    if regenerate or not os.path.isfile(os.path.join(data_dir, "manifest.csv")):
        print("[1/6] Generating synthetic scenes in {} ...".format(data_dir))
        generate(data_dir, seeds)
    else:
        print("[1/6] Reusing synthetic scenes in {} (pass --regenerate to rebuild)".format(data_dir))
    manifest = read_csv(os.path.join(data_dir, "manifest.csv"))
    annotations = read_csv(os.path.join(data_dir, "annotations.csv"))
    config = os.path.join(data_dir, "config.synthetic.json")
    base = Settings.from_json(config)
    work = os.path.join(report_dir, "runs")
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work, mode=0o700)

    print("[2/6] Running {} clips x {} methods through analyze() ...".format(len(manifest), len(METHODS)))
    runs, fps = {}, {}
    for method in METHODS:
        settings = copy.deepcopy(base)
        settings.evidence_dir = os.path.join(work, "evidence")
        settings.trajectory_csv = os.path.join(work, "trajectories_{}.csv".format(method))
        db = os.path.join(work, "{}.db".format(method))
        frames = elapsed = 0.0
        for m in manifest:
            r = analyze(m["video_path"], config, method=method, clip_id=m["clip_id"], scene_id=m["scene_id"],
                        session_id=m["session_id"], split=m["split"], database_path=db,
                        detections=m["video_path"].replace(".mp4", ".tracks.json"), settings=copy.deepcopy(settings), quiet=True)
            frames += r["frames"]; elapsed += r["elapsed_seconds"]
        fps[method] = round(frames / max(elapsed, 1e-3), 1)
        pred_path = os.path.join(work, "{}_predictions.csv".format(method))
        export_db(db, pred_path)
        runs[method] = read_csv(pred_path)
        print("      {:<12} {:>5} alerts  ({} frames/s, detector replay)".format(method, len(runs[method]), fps[method]))

    print("[3/6] Scoring ...")
    results = {
        "core_all": compare_runs(annotations, manifest, runs, os.path.join(report_dir, "comparison_core_all.json"),
                                 split="all", bootstrap=bootstrap, event_types=CORE_EVENTS, quiet=True),
        "core_test": compare_runs(annotations, manifest, runs, os.path.join(report_dir, "comparison_core_test.json"),
                                  split="test", bootstrap=0, event_types=CORE_EVENTS, quiet=True),
        "all_events": compare_runs(annotations, manifest, {k: runs[k] for k in ("context",)}, os.path.join(report_dir, "context_all_events.json"),
                                   split="all", bootstrap=0, quiet=True),
    }
    print(format_table(results["core_all"]))

    print("[4/6] Writing annotated showcase videos ...")
    showcase_dir = os.path.join(report_dir, "showcase"); os.makedirs(showcase_dir, exist_ok=True)
    showcase = []
    test_seed = max(int(m["clip_id"].rsplit("_", 1)[1]) for m in manifest)
    for kind in SHOWCASE:
        clip = "syn_{}_{}".format(kind, test_seed)
        m = next((x for x in manifest if x["clip_id"] == clip), None)
        if not m:
            continue
        for method in ("single_rule", "context"):
            settings = copy.deepcopy(base)
            settings.evidence_dir = os.path.join(work, "showcase_evidence")
            settings.trajectory_csv = os.path.join(work, "showcase_traj.csv")
            out = os.path.join(showcase_dir, "{}_{}.mp4".format(clip, method))
            analyze(m["video_path"], config, method=method, clip_id=clip, database_path=os.path.join(work, "showcase.db"),
                    detections=m["video_path"].replace(".mp4", ".tracks.json"), annotated_out=out, settings=settings, quiet=True)
            showcase.append(out)

    print("[5/6] Labelling trajectories and training the feature baseline ...")
    baseline = None
    try:
        from .label_trajectories import label_rows
        from .train_baseline import train
        import csv
        rows, counts = label_rows(read_csv(os.path.join(work, "trajectories_context.csv")), annotations, manifest)
        labelled = os.path.join(work, "trajectories_labeled.csv")
        with open(labelled, "w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        baseline = train(labelled, os.path.join(report_dir, "baseline_classifier.json"))
        print("      selected {} | test macro-F1 {}".format(baseline["selected_model"], baseline["test"]["macro_f1"]))
    except SystemExit as exc:
        print("      skipped: {}".format(exc))

    print("[6/6] Writing summary ...")
    summary = write_summary(report_dir, manifest, annotations, results, fps, baseline, showcase, seeds)
    print("Done in {:.0f}s. {}".format(time.monotonic() - t0, summary))
    return results


def _row(name, r):
    fa = r["false_alerts_per_camera_hour"]
    return "| {} | {} | {} | {} | {} | {} | {} | {} |".format(name, r["annotation_events"], r["eligible_alerts"],
        r["event_detection_precision"], r["event_detection_recall"], r["event_detection_f1"],
        "{:.1f}".format(fa) if fa is not None else "n/a", r["abstentions"])


def write_summary(report_dir, manifest, annotations, results, fps, baseline, showcase, seeds):
    hours = sum(float(m["duration_seconds"]) for m in manifest) / 3600
    lines = ["# SENTRIX AI synthetic pipeline check", "", "> **{}**".format(BANNER), "",
             "Generated by `python -m sentrix.demo_synthetic`. {} clips ({} scenario types x {} seeds), {:.2f} h of rendered video, {} scripted ground-truth events.".format(
                 len(manifest), len({m['notes'] for m in manifest}), seeds, hours, len(annotations)),
             "Detections come from a noisy simulated tracker, so detector errors (misses on real people, false boxes, ID switches) are under-represented.", "",
             "## Core events (restricted-zone entry, potentially unattended object), all clips", "",
             "| Method | GT events | Alerts | Precision | Recall | F1 | False alerts / cam-hour | Withheld |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    core = results["core_all"]
    for name in ("motion", "single_rule", "context"):
        if name in core: lines.append(_row(name, core[name]))
    boot = core.get("_paired_bootstrap")
    if boot:
        lines += ["", "Paired clip bootstrap ({} iterations) vs `single_rule`:".format(boot["iterations"]), ""]
        for name, d in boot["differences_vs_reference"].items():
            lines.append("- `{}`: F1 difference 95% interval {}, false alerts/hour difference {}".format(name, d["event_f1_delta_95ci"], d["false_alerts_per_hour_delta_95ci"]))
    lines += ["", "![core comparison](comparison_core_all.png)", "", "## Core events, held-out test seed only", "",
              "| Method | GT events | Alerts | Precision | Recall | F1 | False alerts / cam-hour | Withheld |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("motion", "single_rule", "context"):
        if name in results["core_test"]: lines.append(_row(name, results["core_test"][name]))
    ctx = results["all_events"]["context"]
    lines += ["", "## Context method, every event class (incl. optional extensions), all clips", "",
              "| Event | GT | Matched | False alerts | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for event, e in ctx["events"].items():
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(event, e["ground_truth_events"], e["matched_alerts"], e["false_alerts"], e["precision"], e["recall"], e["f1"]))
    lines += ["", "Low-visibility intervals: {}, appropriate abstention rate: {}. Abstentions outside annotated events: {}.".format(
        ctx["low_visibility_intervals"], ctx["appropriate_abstention_rate"], ctx["abstentions_outside_annotated_events"])]
    if ctx.get("breakdown"):
        lines += ["", "Breakdown by recorded condition (context, all events):", ""]
        for field, table in ctx["breakdown"].items():
            lines.append("- **{}**: ".format(field) + "; ".join("{} recall {} / {} false alerts".format(k, v["recall"], v["false_alerts"]) for k, v in table.items()))
    lines += ["", "Pipeline speed with detector replay (excludes YOLO inference): " + ", ".join("{} {} fps".format(k, v) for k, v in fps.items()) + ".", ""]
    if baseline:
        lines += ["## Feature-based sequence baseline (scikit-learn)", "",
                  "Windows: {} train / {} development / {} test. Selected `{}`; test macro-F1 {}.".format(
                      baseline["windows"]["train"], baseline["windows"]["development"], baseline["windows"]["test"], baseline["selected_model"], baseline["test"]["macro_f1"]), "",
                  "| Class | Support | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"]
        for c, v in baseline["test"]["per_class"].items():
            lines.append("| {} | {} | {} | {} | {} |".format(c, v["support"], v["precision"], v["recall"], v["f1"]))
    lines += ["", "## What this shows (and does not)", "",
              "- The pipeline runs end to end: events, evidence gate, SQLite store, blurred evidence clips, export, scoring, comparison, features, classifier.",
              "- Context rules suppress the scripted hard negatives (boundary lingering, owner staying beside a bag) that the single fixed rule flags.",
              "- The evidence gate trades recall for caution: the dim, distant entry is withheld for review instead of alerted.",
              "- Synthetic scenes were written by the developers alongside the rules, so agreement is expected. Real thresholds, recall and false-alert rates must come from consented footage with a frozen held-out split.",
              "", "Annotated videos (face-region blur applied): " + ", ".join("`showcase/{}`".format(os.path.basename(s)) for s in showcase)]
    path = os.path.join(report_dir, "SUMMARY.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/synthetic"); p.add_argument("--reports", default="reports/synthetic")
    p.add_argument("--seeds", type=int, default=5); p.add_argument("--regenerate", action="store_true")
    p.add_argument("--bootstrap", type=int, default=500)
    a = p.parse_args()
    run(a.data, a.reports, a.seeds, a.regenerate, a.bootstrap)

if __name__ == "__main__": main()
