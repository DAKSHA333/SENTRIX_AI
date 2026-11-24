# SENTRIX AI capstone report

## Project summary

SENTRIX AI is a controlled research prototype for explaining selected observable video events to a human reviewer. It does not identify people, infer intent, or automate enforcement.

## Problem, hypothesis, and scope

State the two core events, any implemented extensions, H1/H0, camera setting, study period, and exact exclusions. Separate completed work from proposed work.

## System and implementation

Document detector/model weights and license, tracker, event logic, zone configuration, evidence gate, dashboard roles, privacy blur, retention settings, hardware, package versions, and source revision.

## Dataset and ethics

| Item | Result |
|---|---|
| Consented staged clips | Pending data collection |
| Independent sessions / scenes | Pending |
| Normal camera hours | Pending |
| Core event counts by class | Pending |
| Train / development / test group counts | Pending |
| Consent, site approval, and retention evidence | Pending |

Summarize annotation protocol, hard negatives, visibility metadata, dual-review subset and agreement. Never include names or identifying screenshots in the report.

## Experimental results

| Method | Event | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 | False alerts / camera-hour | Median / p95 delay | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Motion baseline | Pending | | | | | | | | | |
| Single rule | Pending | | | | | | | | | |
| Context-aware | Pending | | | | | | | | | |

Do not populate targets as results. Include all denominators and held-out session/scene counts. Add error breakdowns by lighting, crowd density, occlusion, and viewpoint. Report failed runs and exclusions.

## Software verification (not results)

State that the pipeline was verified end to end with `python -m sentrix.demo_synthetic` and `python -m pytest`, and that synthetic numbers are excluded from the results above. Optionally show one synthetic showcase video to illustrate single-rule vs context behaviour, labelled as synthetic.

## Findings and limitations

Discuss false alerts, misses, tracking switches, abstention, explanation review, compute limits, label coverage, dataset shift, camera-specific thresholds, and confidence interval limitations. State explicitly that a controlled study does not establish safety benefit or deployment readiness.

## Privacy, safeguards, and misuse boundaries

Document consent/site permission, access roles, face-blur review, raw data minimization, audit trail, retention/deletion, and the human confirmation requirement. State that the model must not be used for autonomous detention, confrontation, identity profiling, predictive policing, or claims of guaranteed harm prevention.

## Reproduction

List exact commands, configs, model weight hashes, annotation/manifest versions, hardware, and report output paths.
