"""Feature-based sequence baselines (scikit-learn) for labelled trajectory windows.

This is the "simpler feature-based baseline" the proposal compares the optional GRU
against (section 6.3). Each window of a person track is summarised by interpretable
statistics (speed, displacement, straightness, direction reversals, alignment with
local flow, zone occupancy...) and classified with logistic regression and a random
forest. Models are fitted on the train split only, selected on development, and
reported once on test. Metrics are window-level, not event-level.
"""
import argparse
import json
import os
from math import cos, hypot, sin
import numpy as np
import pandas as pd

SUMMARY_NAMES = ["speed_mean", "speed_std", "speed_max", "net_displacement_norm", "path_length_norm", "straightness",
                 "direction_reversals", "heading_change_mean", "flow_alignment_mean", "crowd_density_mean",
                 "nearest_bag_min", "in_zone_fraction", "box_h_mean", "dwell_last", "pose_arms_raised_mean"]


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def window_features(group):
    x, y = _num(group["x_norm"]).to_numpy(), _num(group["y_norm"]).to_numpy()
    speed = _num(group["speed_px_s"]).fillna(0).to_numpy()
    steps = np.hypot(np.diff(x), np.diff(y))
    path = float(np.nansum(steps)); net = float(hypot(x[-1]-x[0], y[-1]-y[0]))
    # Reversals from displacement over 3-sample strides (robust to box jitter).
    vec = [(x[i+3]-x[i], y[i+3]-y[i]) for i in range(0, len(x)-3, 3)]
    vec = [v for v in vec if hypot(*v) > 0.01]
    rev = sum(1 for a, b in zip(vec, vec[1:]) if (a[0]*b[0]+a[1]*b[1]) / (hypot(*a)*hypot(*b)) < -0.5)
    heading = _num(group["heading_rad"]).fillna(0).to_numpy()
    dh = np.abs(np.angle(np.exp(1j*np.diff(heading)))) if len(heading) > 1 else np.array([0.0])
    fx, fy = _num(group["local_flow_x"]).fillna(0).to_numpy(), _num(group["local_flow_y"]).fillna(0).to_numpy()
    align = np.array([cos(h)*a + sin(h)*b for h, a, b in zip(heading, fx, fy)])
    bag = _num(group["nearest_bag_distance_norm"])
    zones = group["zone_names"].fillna("").astype(str)
    arms = _num(group.get("pose_arms_raised", pd.Series(dtype=float))).fillna(0)
    return [float(np.mean(speed)), float(np.std(speed)), float(np.max(speed)), net, path, net / path if path > 0 else 0.0,
            float(rev), float(np.mean(dh)), float(np.mean(align)), float(_num(group["crowd_density_per_mpix"]).fillna(0).mean()),
            float(bag.min()) if bag.notna().any() else 1.0, float((zones != "").mean()), float(_num(group["box_h_norm"]).fillna(0).mean()),
            float(_num(group["dwell_seconds"]).fillna(0).iloc[-1]), float(arms.mean()) if len(arms) else 0.0]


def build_windows(data, window=20, stride=5):
    X, y, split, groups = [], [], [], []
    data = data.copy()
    data["timestamp_seconds"] = _num(data["timestamp_seconds"])
    for (clip, track), g in data.sort_values("timestamp_seconds").groupby(["clip_id", "track_id"]):
        if len(g) < window:
            continue
        for start in range(0, len(g) - window + 1, stride):
            w = g.iloc[start:start+window]
            X.append(window_features(w)); y.append(str(w["label"].iloc[-1]))
            split.append(str(w["split"].iloc[-1]).lower()); groups.append(clip)
    return np.asarray(X, dtype="float64"), np.asarray(y), np.asarray(split), np.asarray(groups)


def per_class(y_true, y_pred, classes):
    out = {}
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum()); fp = int(((y_true != c) & (y_pred == c)).sum()); fn = int(((y_true == c) & (y_pred != c)).sum())
        p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
        out[c] = {"support": int((y_true == c).sum()), "precision": round(p, 4), "recall": round(r, 4), "f1": round(2*p*r/(p+r), 4) if p+r else 0.0}
    return out


def train(features_csv, out, window=20, stride=5, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    data = pd.read_csv(features_csv)
    missing = {"clip_id", "track_id", "timestamp_seconds", "label", "split"} - set(data.columns)
    if missing:
        raise SystemExit("Feature CSV is missing columns: {} (run sentrix.label_trajectories first)".format(sorted(missing)))
    data = data[data["label"].notna() & (data["label"].astype(str) != "")]
    X, y, split, groups = build_windows(data, window, stride)
    train_m, dev_m, test_m = split == "train", np.isin(split, ["development", "dev"]), split == "test"
    if train_m.sum() < 20 or test_m.sum() < 1 or len(set(y[train_m])) < 2:
        raise SystemExit("Insufficient labelled windows (train={}, test={}, train classes={}). Collect more independent sessions.".format(
            int(train_m.sum()), int(test_m.sum()), sorted(set(y[train_m]))))
    leak = set(groups[train_m]) & set(groups[test_m])
    if leak:
        raise SystemExit("Clip(s) appear in both train and test splits: {}".format(sorted(leak)[:5]))
    classes = sorted(set(y))
    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed),
    }
    report = {"window_samples": window, "stride_samples": stride, "features": SUMMARY_NAMES, "classes": classes,
              "windows": {"train": int(train_m.sum()), "development": int(dev_m.sum()), "test": int(test_m.sum())},
              "class_counts_train": {c: int((y[train_m] == c).sum()) for c in classes}, "models": {}}
    select_on = dev_m if dev_m.any() else train_m
    best, best_score = None, -1
    for name, model in candidates.items():
        model.fit(X[train_m], y[train_m])
        score = f1_score(y[select_on], model.predict(X[select_on]), average="macro", zero_division=0)
        report["models"][name] = {"selection_macro_f1": round(float(score), 4), "selection_split": "development" if dev_m.any() else "train"}
        if score > best_score:
            best, best_score = name, score
    model = candidates[best]
    pred = model.predict(X[test_m])
    report["selected_model"] = best
    report["test"] = {"macro_f1": round(float(f1_score(y[test_m], pred, average="macro", labels=classes, zero_division=0)), 4),
                      "per_class": per_class(y[test_m], pred, classes),
                      "confusion_matrix": {"labels": classes, "rows_true_cols_pred": confusion_matrix(y[test_m], pred, labels=classes).tolist()}}
    if best == "random_forest":
        imp = model.feature_importances_
        report["feature_importance"] = {n: round(float(v), 4) for n, v in sorted(zip(SUMMARY_NAMES, imp), key=lambda t: -t[1])}
    report["caution"] = "Window-level classification on the given split only; not event-level detection and not deployment performance."
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("features_csv", help="Labelled trajectory CSV from sentrix.label_trajectories")
    p.add_argument("--out", default="reports/baseline_classifier.json")
    p.add_argument("--window", type=int, default=20, help="Samples per window")
    p.add_argument("--stride", type=int, default=5)
    a = p.parse_args()
    print(json.dumps(train(a.features_csv, a.out, a.window, a.stride), indent=2))

if __name__ == "__main__": main()
