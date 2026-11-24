# SENTRIX evaluation protocol

## Frozen comparison

Run three methods on identical held-out clips, camera zones, and hardware:

1. `motion`: frame-difference activity baseline.
2. `single_rule`: detector/tracker plus one fixed event rule without context evidence gating, schedules, or optional extensions.
3. `context`: configured event logic, context features, and insufficient-evidence gate.

Freeze class thresholds and zone geometry using training/development sessions before opening held-out labels. Record model weights/version, package versions, config hash, camera resolution/FPS, compute device, and every excluded clip.

If a zone has an `active_hours` schedule, document the site-entered schedule and timezone. For recorded video, pass the original start timestamp with `--recorded-start`; without it, the schedule gate abstains rather than assuming a schedule match.

## Baseline definitions

- `motion` (A): frame-difference area alarm with a cooldown; untyped.
- `single_rule` (B): one fixed rule per core event: the first sampled boundary crossing (no confirmation samples), and a bag stationary for `unattended_seconds` regardless of nearby people. No schedules, extensions or evidence gate.
- `context` (C): foot-point zone test confirmed over `zone_entry_confirm_samples`, bag stationary **and** previously nearby person tracks moved beyond `association_distance_px` for `separation_seconds`, schedules, optional extensions, and the evidence gate (reason codes in `evidence_reasons`).

Context ablations for the report: rerun `context` with `zone_entry_confirm_samples: 1`, with the gate thresholds relaxed (for example `low_quality_min_confidence: 0`, `occlusion_iou: 0`, `low_light_brightness: 0`), and with `separation_seconds: 0`, to attribute the gains separately.

## Metrics

Match one alert to at most one annotated event using the predeclared time tolerance (default two seconds) and event class. Report per-event precision, recall, F1, counts, Wilson 95% intervals, macro-F1 for typed classifiers, overall event-detection precision/recall/F1 including all unmatched alerts, unmatched alerts per evaluated camera-hour with the hour denominator, median and 95th percentile delay after `trigger_seconds`, correct abstention share for low-visibility intervals, track ID switches if identity tracks are annotated, and measured processing FPS. Treat `motion_alarm` as an untyped baseline candidate that may match a temporally overlapping labeled event; it is not event classification, so macro-F1 is not defined for that baseline.

Review misses, false alerts, ID switches, and abstentions by illumination, crowd density, occlusion, and viewpoint. Report scene/session counts. Per-event Wilson intervals do not account for scene clustering; do not overstate generalization from a small benchmark. `sentrix.compare` reports a clip-level paired bootstrap interval for the F1 and false-alerts/hour differences against the reference method; with few scenes this is optimistic, so also report scene/session counts. Use `--events core` for the H1 comparison and `benchmark.py` breakdowns for illumination, crowd density and viewpoint.

Use `python -m sentrix.annotation_agreement ANNOTATIONS.csv --reviewer-a ID1 --reviewer-b ID2` for event-match agreement on the dual-reviewed subset. If frame-level person tracks are labeled and associated with the output, run `python -m sentrix.tracking_metrics TRACKS.csv` for ID-switch counts.

## Reproducibility commands

Run each method against the same manifest-listed clips with `python -m sentrix.analyze VIDEO --method motion|single_rule|context --config CONFIG`. Export each run's database with `python -m sentrix.export --database DB --out predictions.csv`, then evaluate using `python -m sentrix.benchmark --annotations data/benchmark/annotations.csv --manifest data/benchmark/manifest.csv --predictions predictions.csv --out reports/evaluation.json`.

The 0.80 macro-F1 and false-alarm goals are proposal targets, not achieved results. Publish measured values only after the held-out run and failure review.
