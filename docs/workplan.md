# SENTRIX capstone workplan

This maps the proposal's 80-hour SIC allocation to the current workspace. Status is implementation status only; it does not claim team hours were spent or research data was collected.

| Proposal phase | Planned hours | Workspace deliverables | Remaining study work |
|---|---:|---|---|
| Audit, scope, ethics | 9 | Initial code audit, scope boundary, consent checklist, model card | Confirm team prior assets, site permission, institutional/data-protection review |
| Data and annotation | 16 | Manifest/annotation templates, group split validation, annotation and agreement protocol | Collect and annotate 100–120 consented staged clips, dual-review subset |
| CNN and tracking | 16 | YOLO person/bag/object inference, ByteTrack integration with IoU fallback, scene-local IDs, feature export, detections replay, annotated blurred output video | Install `lap` for ByteTrack; tune camera thresholds; annotate tracks if measuring ID switches |
| Contextual event logic | 16 | Core events, optional dwell/pacing, counter-flow, dispersion, schedules, zone hysteresis and foot-point anchor, evidence gate with reason codes, distinct single-rule baseline, optional pose/GRU, scikit-learn sequence baseline | Validate thresholds on development data; train GRU only if sequence volume supports it |
| Dashboard and tests | 14 | Local Streamlit review with summary counts and evidence flags, FastAPI, evidence clips/face blur, reviewer decisions/ratings, audit schema, 31 unit/integration tests, synthetic end-to-end demo | Configure passwords, inspect blur quality, conduct explanation usefulness review |
| Documentation and demo | 9 | Reproduction workflow, evaluation pipeline with per-clip matching, condition breakdowns and paired bootstrap, report template, run manifest, controlled live demo, implementation audit (docs/audit.md) | Run three methods on frozen held-out scenes; fill report with actual metrics and failure analysis |

The 100–120 clip goal and 80-hour plan are proposal targets. Optional five-class/10-background-hour scope remains stretch work. No held-out performance result is claimed in this repository.

## SIC syllabus mapping

- NumPy/Pandas: trajectory cleaning, feature tables, benchmark reports.
- Supervised learning and metrics: logistic regression / random forest window classifier, optional GRU sequence experiment; event precision/recall/F1, Wilson intervals and paired bootstrap.
- Unsupervised learning: development-split movement clustering for exploration only.
- CNN: pretrained person/object detection.
- RNN/GRU: optional temporal feature classifier when labels and sequence counts suffice.
- Probability/statistics: confidence abstention, interval estimates, latency and false-alert denominators.
