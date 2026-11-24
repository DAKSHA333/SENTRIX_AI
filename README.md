# SENTRIX AI

SENTRIX AI is a controlled research prototype that flags configured, observable video events for human review. It does not identify people, infer criminal intent, or guarantee that harm will be prevented. Knife/scissors alerts are an added user-requested feature and are not a claim in the capstone proposal.

**Status:** software for every proposal phase is implemented and tested (43 unit/integration tests), plus a live demo website. What remains is the study itself: consented footage, annotation, per-camera threshold tuning on train/development sessions, and one frozen held-out run. See [docs/audit.md](docs/audit.md) for the module-by-module ledger.

## Quick start: see the whole pipeline work in 2 minutes (no camera, no model download)

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                 # 43 tests
python -m sentrix.demo_synthetic    # synthetic scenes -> 3 methods -> scoring -> reports/synthetic/SUMMARY.md
```

The synthetic demo renders scripted scenes (rectangle "people" and "bags"), simulates a noisy detector/tracker, and runs them through the real `analyze()` pipeline, evidence clips, SQLite store, export, benchmark, paired bootstrap comparison, trajectory labelling and the scikit-learn baseline. It also writes face-blurred annotated videos to `reports/synthetic/showcase/` that show single-rule vs context behaviour side by side. **It verifies software behaviour only. Never report synthetic numbers as results.**

To browse the synthetic alerts in the reviewer dashboard:

```sh
SENTRIX_DB_PATH=reports/synthetic/runs/context.db streamlit run dashboard.py --server.address 127.0.0.1
```

## Live demo website (front camera → alerts → notifications)

```sh
source .venv/bin/activate
python -m sentrix.live --camera 0 --config config.live.json --mirror
```

Your browser opens **http://127.0.0.1:8080**: a live annotated feed, an alert feed with face-blurred snapshots, Confirm / Dismiss / Escalate buttons, an alarm sound, and browser pop-up notifications (click **Enable browser notifications** once). The first run downloads `yolov8s.pt` (~22 MB) and `yolov8n-pose.pt` (~6 MB). On a Mac, allow camera access for Terminal or Antigravity (System Settings → Privacy & Security → Camera). If the frame rate is low, set `"model": "yolov8n.pt"` in `config.live.json`. Camera `1` may be an iPhone (Continuity Camera); use `--camera 0` for the built-in one.

### What it watches for (config.live.json)

| # | Detector | How to show it on the front camera |
|---|---|---|
| 1 | Knife | Hold a kitchen knife (blade visible, side-on) in front of the camera for ~1 s |
| 2 | Scissors | Hold scissors open in view |
| 3 | Bat / club | Hold a cricket or baseball bat in view (model class "baseball bat") |
| 4 | Gun / firearm | Needs a custom model: see *Adding gun detection* below |
| 5 | Unattended bag | Put a backpack down in view, walk out of frame, wait 15 s |
| 6 | Restricted-zone entry | Step into the right-hand third of the picture (orange zone); recalibrate with `python -m sentrix.calibrate --camera 0 --config config.live.json --zone-name demo_restricted_zone` |
| 7 | Loitering | Stay inside the zone for 15 s |
| 8 | Camera tampering | Cover the lens with your hand for 2 s |
| 9 | Rapid movement | Run or move quickly across the view |
| 10 | Crowding | Three or more people in view for 3 s |
| 11 | Both arms raised | Raise both hands above your shoulders for 2 s |

An object must appear in 3 consecutive detections before alerting, and the same object type alerts at most once every 20 s, which stops flicker from spamming people. Detections that fail the evidence checks (too small, too dark, low confidence, overlapping people) appear on the website as **Low evidence** and are not pushed to phones. Escalate re-sends the alert to every contact.

### Send alerts to the right people

Copy `notify.example.json` to `notify.json`, keep the channels you want, then test:

```sh
cp notify.example.json notify.json
python -m sentrix.notify --config notify.json --test
```

- **Telegram (recommended, free, arrives with the photo):** in Telegram talk to `@BotFather` → `/newbot` → copy the token into `bot_token`. Every person who should receive alerts opens your bot and presses Start. Then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and copy each `"chat":{"id": ...}` into `chat_ids`. A group works too: add the bot to the group and use the group's (negative) id.
- **Email (Gmail):** turn on 2-step verification, create an *App password* (Google Account → Security → App passwords) and use it as `password` with `smtp.gmail.com`, port 587. Put every recipient in `to`.
- **Webhooks:** any URL that accepts a JSON POST (Slack/Discord/Teams incoming webhooks, your own server).

`notify.json` holds secrets and is git-ignored. Notifications run in the background; a failed send is shown in the website's Notifications panel and never stops monitoring.

### Adding gun detection

The standard COCO model has **no firearm class**, so guns (real or toy) are not detected out of the box. To add one, train or obtain a YOLOv8 model with firearm classes and point `weapon_model` at it:

```sh
# Example: a YOLOv8-format firearm dataset (e.g. exported from Roboflow Universe) with data.yaml
yolo detect train data=path/to/data.yaml model=yolov8n.pt epochs=50 imgsz=640
# then in config.live.json:  "weapon_model": "runs/detect/train/weights/best.pt"
```

The model's own class names (for example `pistol`, `rifle`) are added to the alert list automatically. Check the dataset licence, and evaluate it on your own held-out clips before claiming any accuracy.

### Wording

Alerts name what the camera saw ("Possible knife detected", "Camera view obstructed"), never a person or a type of person, and every alert asks for human review. That matches the proposal's boundary: the system flags observable objects and events; it does not identify thieves or terrorists or predict intent.

## Implemented scope

- Recorded-video and permissioned live-camera analysis; YOLO person/bag/object detection with scene-local ByteTrack IDs (IoU tracker fallback), or replay of pre-computed tracks (`--detections`).
- Core events: restricted-zone entry and potentially unattended bag-like object.
- Optional extensions: prolonged dwell / repeated path reversals, counter-flow against coherent local movement, sudden outward group dispersion.
- Three comparison modes that match proposal section 13.1:

| Mode | Zone entry | Unattended object | Extensions | Evidence gate |
|---|---|---|---|---|
| `motion` (Baseline A) | frame-difference area alarm | — | — | — |
| `single_rule` (Baseline B) | first boundary crossing | bag stationary ≥ T | off | off |
| `context` (Proposed C) | foot-point crossing confirmed for N samples, schedules | stationary ≥ T **and** nearby person moved away ≥ S | on if enabled | on |

- Evidence gate: withheld alerts become `insufficient_evidence` with machine-readable reasons: `LOW_DETECTOR_CONFIDENCE`, `TARGET_TOO_SMALL`, `HEAVY_OVERLAP_WITH_OTHER_PERSON`, `LOW_LIGHT`, `IMAGE_BLURRED` (opt-in), `OBJECT_TRACK_FRAGMENTED`, `PERSON_ASSOCIATION_AMBIGUOUS`, `SCHEDULE_TIME_UNAVAILABLE`.
- Template explanations built from the measured values (durations, distances, sample counts), never free-form text.
- Short evidence clips and optional annotated output video with face-region blur; local reviewer decisions, explanation ratings and an audit log in SQLite.
- Benchmark tooling: group-safe splits, validation, event matching per clip, Wilson intervals, false alerts per camera-hour, delay, abstention scoring, breakdown by illumination / crowd density / viewpoint, clip-level paired bootstrap for H1.
- Learning components: trajectory feature export, label joiner, scikit-learn sequence baseline (logistic regression / random forest), optional TensorFlow GRU ablation, exploratory k-means.

## Set up a real camera or clip

Calibrate the zone where people's **feet** touch the floor (zones are tested at the bottom-centre of each person box):

```sh
python -m sentrix.calibrate --camera 0 --config config.demo.json                       # live camera (f freezes the view)
python -m sentrix.calibrate --video clip.mp4 --frame 50 --config config.research.json --zone-name storage_door
```

Click at least three corners, `u` undo, `r` reset, `s` save. Then run:

```sh
python -m sentrix.analyze 0 --live --config config.demo.json --clip-id demo1
python -m sentrix.analyze clip.mp4 --config config.demo.json --annotated-out reports/clip_annotated.mp4 --show
```

Press `q` in the preview window to stop. `config.demo.json` uses lower thresholds for a controlled presentation; never report metrics from that profile. An authorized RTSP URL can be supplied with `--stream URL --live`. YOLO weights download on first use.

## Run the research pipeline

Copy `config.research.json`, configure camera-specific zones and thresholds, and assign anonymous IDs and frozen splits. Use a separate database per method:

```sh
for M in motion single_rule context; do
  python -m sentrix.analyze clip001.mp4 --config config.research.json --method $M --database data/$M.db \
      --clip-id clip001 --scene-id sceneA --session-id session01 --split test
done
```

Process identical footage with identical zones and hardware. `processing_fps` is printed at the end. `data/trajectories.csv` contains anonymous within-run derived features; it may still be personal data and needs restricted access and retention.

## Build a labelled benchmark

```sh
python -m sentrix.dataset init data/benchmark
python -m sentrix.dataset split --manifest data/benchmark/manifest.csv
python -m sentrix.dataset validate --manifest data/benchmark/manifest.csv --annotations data/benchmark/annotations.csv --check-paths
```

Fill `manifest.csv` and `annotations.csv` using [the annotation protocol](docs/annotation_protocol.md). Target 100–120 short core-event and benign-counterexample clips, plus normal footage hours. Obtain site permission and written consent first ([checklist](docs/consent_checklist.md)). The split command needs at least three independent scene/session groups.

## Evaluation and reporting

```sh
for M in motion single_rule context; do python -m sentrix.export --database data/$M.db --out reports/${M}_predictions.csv; done
python -m sentrix.benchmark --annotations data/benchmark/annotations.csv --manifest data/benchmark/manifest.csv \
    --predictions reports/context_predictions.csv --events core --out reports/context_metrics.json
python -m sentrix.compare --annotations data/benchmark/annotations.csv --manifest data/benchmark/manifest.csv \
    --run motion=reports/motion_predictions.csv --run single_rule=reports/single_rule_predictions.csv \
    --run context=reports/context_predictions.csv --events core --reference single_rule --out reports/method_comparison.json
```

`--events core` restricts scoring to the two core events (H1). `--split development` scores the tuning split; the default is `test`. `compare` writes a chart and a paired clip bootstrap of the F1 and false-alerts/hour differences against the reference method.

Learning experiments on reviewed trajectories:

```sh
python -m sentrix.label_trajectories data/trajectories.csv --annotations data/benchmark/annotations.csv --manifest data/benchmark/manifest.csv --out data/trajectories_labeled.csv
python -m sentrix.train_baseline data/trajectories_labeled.csv --out reports/baseline_classifier.json
python -m sentrix.train_gru data/trajectories_labeled.csv --out reports/gru_metrics.json          # needs TensorFlow
python -m sentrix.cluster_movement data/trajectories_labeled.csv --split development --out reports/movement_clusters.csv
```

Other tools: `python -m sentrix.annotation_agreement ANNOTATIONS.csv --reviewer-a r1 --reviewer-b r2`, `python -m sentrix.tracking_metrics TRACKS.csv` (ID switches), `python -m sentrix.run_manifest config.research.json` (config/weight hashes and package versions). Use [the evaluation protocol](docs/evaluation_protocol.md) and fill the [SIC report template](docs/sic_report_template.md) with measured results only.

## Review alerts and protect data

```sh
export SENTRIX_DB_PATH=data/sentrix_research.db
export SENTRIX_REVIEWER_PASSWORD='choose-a-local-reviewer-password'
export SENTRIX_ADMIN_PASSWORD='choose-a-different-admin-password'
streamlit run dashboard.py --server.address 127.0.0.1
```

The dashboard shows review counts, an alert-type chart, filters, evidence-quality flags, blurred clips, confirm/dismiss/escalate with notes, explanation ratings, and (admin) the audit history. Without passwords it opens in local-demo mode with a warning; use localhost only. Automated face blur can miss faces and needs manual review. If the OpenCV build lacks the Haar cascade file, only person-box head regions are blurred (a warning is printed).

The optional local API needs bearer tokens:

```sh
export SENTRIX_REVIEWER_TOKEN='choose-a-review-token' SENTRIX_ADMIN_TOKEN='choose-a-different-admin-token'
uvicorn sentrix.api:app --host 127.0.0.1 --port 8000
```

Endpoints: `/health`, `GET /alerts`, `GET /alerts/{id}`, `POST /alerts/{id}/decision`, admin `GET /audit`. Retention: `python -m sentrix.retention --config config.research.json --dry-run`, then without `--dry-run`.

## Verification status and limits

Unit and integration tests cover zone hysteresis and foot-point anchoring, the evidence gate and each reason, context vs single-rule unattended logic, extension robustness to jitter, state pruning, per-clip matching, abstention scoring, event filtering, bootstrap, trajectory labelling, and an end-to-end synthetic run through analyze → store → evidence → export → benchmark. The YOLO/ByteTrack path, Streamlit dashboard and FastAPI server need their packages installed and were not exercised in the build environment. No consented footage or annotations are included; camera-specific thresholds, face-blur quality, FPS and the proposal targets are not established. See [the model card](docs/model_card.md).
