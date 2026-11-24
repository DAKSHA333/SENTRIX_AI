# SENTRIX implementation audit and progress ledger

Proposal Step 1 asks for each module to be classified as completed, partially working or planned. This ledger covers software only; it does not claim team hours, data collection, or results.

**Legend:** ✅ implemented and tested here · 🟡 implemented, needs packages or hardware not available in the build environment · ⏳ study work (needs consented data)

## Changes made during completion (November 2026)

| Area | Problem found | Fix |
|---|---|---|
| Benchmark | Alerts were matched to ground truth from **any clip** at a similar timestamp, inflating recall/precision on multi-clip test sets | Matching is now restricted to the same `clip_id` |
| Benchmark | `appropriate_abstention_rate` crashed (`abstention_hits` undefined) whenever a low-visibility annotation existed | Fixed; also reports abstentions outside annotated events |
| Baseline B | `single_rule` differed from `context` only by the evidence gate, so the H1 comparison was mostly a gate ablation | `single_rule` is now one fixed rule per event (first crossing; bag stationary ≥ T); context adds hysteresis, person-separation context, schedules, extensions and the gate |
| Zone entry | Box centre (torso) was tested against floor polygons; one jittery sample on the boundary produced an alert | Foot-point anchor (`zone_anchor`), confirmation over N consecutive samples (`zone_entry_confirm_samples`) |
| Evidence gate | Only confidence and box size were checked; proposal also names occlusion, ambiguous association, interrupted tracks and image quality | Added overlap, low light, optional blur, fragmented object track and ambiguous association, each returned as a reason code |
| Pacing | History older than the window silently disabled pacing; per-sample direction flips counted box jitter as reversals | Window filtering; displacement-based reversals (`pacing_min_step_px`) |
| Counter-flow | Raw velocity sum let one fast walker define "the crowd"; no reliability check | Unit-vector mean with a coherence threshold, 1 s smoothed velocity |
| Dispersion | `dispersion_since or now` reset the timer when the event started at t = 0 | Explicit `None` check |
| Memory | Per-track state grew without bound in long live runs; 3 s pre-roll of 1080p frames ≈ 0.5 GB | State TTL pruning; evidence frames downscaled to `evidence_max_width` |
| Config | Misspelled keys silently fell back to defaults; demo polygon had each corner clicked twice (self-intersecting) | Unknown keys rejected (keys starting with `_` allowed as comments); demo polygon repaired |
| Calibration | Live camera only, replaced zone 0 only | Works on recorded clips (frozen frame), named zones, undo, freeze |
| Analyze | Required ultralytics even for the motion baseline; no way to produce a demo video of a recorded clip | Lazy YOLO import, `--detections` replay, `--annotated-out` (face-blurred), `--show`, `--max-frames` |

## Live demo additions

| Addition | Module(s) | Status |
|---|---|---|
| Live monitor + local website (MJPEG feed, alert feed, review buttons, browser notifications, alarm) | `live.py`, `web/index.html` | ✅ tested with replayed tracks; 🟡 real camera needs ultralytics |
| Notifications: Telegram (with snapshot), email, webhooks, cooldown, escalation | `notify.py` | ✅ (tested with a fake transport) |
| Object alerts with confirmation + per-label cooldown; knife, scissors, bat | `events.py` | ✅ |
| Optional custom weapon model (e.g. firearm) | `analyze.WeaponDetector` | 🟡 needs a trained model |
| Camera tampering, crowding, rapid movement, both arms raised | `events.py`, `analyze.SceneMonitor` | ✅ (arms raised needs pose weights) |

## Module ledger

| Proposal item | Module(s) | Status |
|---|---|---|
| CNN detection | `analyze.py` (YOLO) | 🟡 needs `ultralytics` + weights |
| Multi-object tracking | `analyze.py` ByteTrack adapter, `tracking.py` IoU fallback | 🟡 ByteTrack needs `lap`; IoU tracker ✅ |
| Core event: restricted-zone entry | `events.py` | ✅ |
| Core event: potentially unattended object | `events.py` | ✅ |
| Extension: dwell / repeated reversals | `events.py` | ✅ |
| Extension: counter-flow | `events.py` | ✅ |
| Extension: group dispersion | `events.py` | ✅ |
| Context rules, schedules, hysteresis | `events.py`, `config.py` | ✅ |
| INSUFFICIENT_EVIDENCE gate with reasons | `events.py`, `analyze.frame_quality` | ✅ |
| Optional pose cues | `pose.py` | 🟡 needs pose weights |
| Motion baseline (A) / single rule (B) / context (C) | `baselines.py`, `events.py` | ✅ |
| Evidence clips with face blur | `evidence.py` | ✅ (blur quality must be checked by eye on real footage) |
| Reviewer dashboard, decisions, ratings, audit | `dashboard.py`, `store.py` | 🟡 needs `streamlit`; storage ✅ |
| Local API | `api.py` | 🟡 needs `fastapi` |
| Retention | `retention.py` | ✅ |
| Dataset manifest, splits, validation | `dataset.py` | ✅ |
| Event metrics, false alerts/hour, delay, abstention, breakdowns | `benchmark.py` | ✅ |
| Method comparison + paired bootstrap (H1) | `compare.py` | ✅ |
| Feature-based sequence baseline | `train_baseline.py`, `label_trajectories.py` | ✅ |
| GRU ablation | `train_gru.py` | 🟡 needs TensorFlow and enough labelled sequences |
| Unsupervised exploration | `cluster_movement.py` | ✅ |
| Annotation agreement, ID switches, run manifest | `annotation_agreement.py`, `tracking_metrics.py`, `run_manifest.py` | ✅ |
| Synthetic end-to-end verification | `simulate.py`, `demo_synthetic.py` | ✅ |
| Consented staged benchmark (100–120 clips) | — | ⏳ |
| Per-camera threshold tuning on train/dev | — | ⏳ |
| Frozen held-out comparison, FPS on target laptop | — | ⏳ |
| Explanation-usefulness reviewer study | dashboard ratings → `benchmark.py` | ⏳ |

## Remaining study checklist (in order)

1. Install requirements on the demo laptop; run `python -m pytest -q` and `python -m sentrix.demo_synthetic`.
2. Run one live camera check: `sentrix.calibrate`, then `sentrix.analyze 0 --live`; note FPS.
3. Complete the consent checklist and record the staged clips (core events and hard negatives).
4. Annotate, double-review a subset, run `dataset split` and `validate`; freeze the split.
5. Tune thresholds on train/development only (`benchmark --split development`).
6. Run all three methods once on test; `compare --events core`; fill the SIC report template with those numbers and the failure analysis.
