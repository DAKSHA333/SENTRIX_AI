# SENTRIX research model card

## Intended use

Controlled, consented capstone research on visible, predefined video events for human review. Not for deployment screening, identity recognition, enforcement, threat prediction, or a guarantee of safety.

## Components

| Component | Current default | Record before each experiment |
|---|---|---|
| Object detector | Ultralytics YOLOv8 nano COCO weights (`yolov8n.pt`) | Exact file hash, package version, license/redistribution terms, class list, confidence and image size |
| Tracker | Ultralytics ByteTrack (`bytetrack.yaml`) | Exact YAML/config hash, package version, ID-switch review |
| Optional pose | YOLOv8 nano pose (`yolov8n-pose.pt`) when enabled | Exact weight hash, license, keypoint visibility, frame-rate impact |
| Feature baseline | scikit-learn logistic regression / random forest over window statistics | Dataset version, split, window length, selected model, test per-class metrics |
| Optional temporal model | GRU over reviewed trajectory features | Dataset version, scene/session split, normalization, class balance, checkpoints, metrics |

COCO labels include broad household-object classes such as knife and scissors; those labels are not a purpose-built weapons taxonomy. Standard COCO weights do not include a firearm class. Do not represent generic object labels as reliable weapon detection.

## Known limits

Small/distant objects, occlusion, low light, unusual viewpoints, camera motion, class imbalance, tracker switches, domain shift, and limited staged data can cause misses and false alerts. Face-region blurring uses automatic image processing and must be manually inspected; it is not a guarantee that every face is blurred. Local IDs are temporary scene tracks and are not identity.

## Reproducibility

Run `python -m sentrix.run_manifest CONFIG --out reports/run_manifest.json` and retain the output beside each metrics report. Confirm model license terms and package versions for the actual downloaded weights before reuse or redistribution.
