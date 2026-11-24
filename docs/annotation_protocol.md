# SENTRIX annotation and data protocol

## Before recording

Use only an authorized, controlled site. Obtain written, informed consent from everyone who will appear; post any required notice; keep a record of site permission, consent, purpose, access list, retention date, and deletion owner. Do not record unsuspecting bystanders. Do not stage threatening, violent, or panic-inducing behavior. Stop collection if anyone withdraws consent.

Use a fixed camera, record resolution/FPS/lighting/viewpoint, and assign random clip/session IDs. Store raw video outside the source repository with restricted permissions. Avoid names and identity labels. Keep the key mapping participants to consent records separately from annotations and delete it on schedule.

## Event labels

- `restricted_zone_entry`: onset is the first frame where the person track crosses the configured polygon boundary. Record zone name and entry time. Authorization is not inferred.
- `potentially_unattended_object`: onset is the first time the predeclared stillness and separation criteria are both satisfied. Record bag-like class, measured stationary duration, nearest-track distance, and association uncertainty. Do not label ownership.
- `prolonged_zone_dwell`: record zone entry and configured dwell trigger time.
- `repeated_path_reversals`: record the event interval and number of reversals under the frozen rule.
- `sustained_counter_flow`: label only when the local flow estimate is reliable; include crowd-track count and the frozen duration criterion.
- `sudden_group_dispersion`: describe outward group motion only; do not label panic or intent.
- Added detector object labels such as `knife`/`scissors` should be separate, frame-level object annotations and must not imply use, threat, or intent.

## Hard negatives and quality tags

Include routine walks near a zone, entering nonrestricted areas, waiting, briefly setting down/retrieving a bag, a bag owner remaining nearby, brief occlusion, poor illumination, crowd overlap, and camera movement where naturally available. Record `low_visibility=true` plus a reason when a human reviewer cannot reliably judge the event. Do not convert ambiguity into a positive label.

The initial study target is 100–120 short core-event and benign-counterexample clips. Three extension event types and the 150-clip/10-background-hour benchmark are stretch goals only. Public anomaly datasets may be used only when the published labels, license, and use conditions match the specific comparison; they do not replace event-specific ground truth.

Example zones can be configured after calibration:

```json
{"name":"restricted_entry","polygon":[[100,100],[400,100],[400,500],[100,500]],"zone_type":"restricted","alert_on_entry":true,"enable_dwell":false,"active_hours":["09:00-17:00"]}
{"name":"waiting_area","polygon":[[500,100],[800,100],[800,500],[500,500]],"zone_type":"waiting","alert_on_entry":false,"enable_dwell":true,"active_hours":[]}
```

Coordinates are camera-specific pixel positions and must be confirmed against the actual view. The second zone's dwell event runs only when the global dwell/pacing option is enabled.

## Annotation and splits

Annotate clip-relative event start/end and `trigger_seconds`. Use at least two reviewers on a randomly selected subset. Resolve disagreements with a written adjudication note and report agreement. Keep all clips from the same scene and recording session in one split; use approximate 60/20/20 group-based train/development/test partitions. Never tune rules or learned thresholds on held-out test scenes.

Create blank CSV templates with `python -m sentrix.dataset init data/benchmark`. Validate metadata and group leakage with `python -m sentrix.dataset validate --manifest data/benchmark/manifest.csv --annotations data/benchmark/annotations.csv`.

## Privacy and retention

For evidence clips, apply face-region blurring before writing. Inspect blur quality manually; automated face detection can miss faces. Keep only short event clips and derived trajectories needed for the study, restrict access, log reviewer decisions, and delete raw footage and clips at the approved retention date. Tracks can remain personal data even without names.
