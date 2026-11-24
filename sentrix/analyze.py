"""Video analysis entry point (recorded clip, camera index, or authorized stream).

Detection backends:
  * YOLO + ByteTrack (default; needs ultralytics, imported lazily)
  * YOLO + IoU tracker   (tracker_backend = "iou")
  * pre-computed tracks  (--detections FILE.json): replays a detections sidecar,
    e.g. from sentrix.simulate, so the full pipeline can run without a model.
"""
import argparse
import json
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import cv2
import numpy as np
from .config import Settings
from .events import EventEngine
from .store import AlertStore
from .tracking import IoUTracker, Track
from .evidence import EvidenceRecorder, blur_faces, load_face_cascade
from .baselines import MotionBaseline
from .features import FeatureLogger
from .pose import extract_pose_features

METHODS = ("motion", "single_rule", "context")


def load_yolo(weights):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. Run `pip install -r requirements.txt`, "
                         "or replay pre-computed tracks with --detections.") from exc
    return YOLO(weights)


def resolve_device(requested):
    """"auto" picks Apple-silicon MPS or CUDA when available, otherwise CPU."""
    if requested not in ("", "auto"):
        return requested
    if requested == "":
        return ""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def device_kw(settings):
    return {"device": settings.device} if settings.device and settings.device != "auto" else {}


class WeaponDetector:
    """Optional second YOLO model for classes COCO lacks (e.g. a custom firearm model).

    Its detections get their own IoU tracker and IDs offset by 1,000,000 so they never
    collide with the main tracker's IDs. Class names come from the custom model.
    """
    def __init__(self, settings):
        self.model = load_yolo(settings.weapon_model)
        self.settings = settings
        self.tracker = IoUTracker(settings.max_track_gap, settings.match_iou)
        self.labels = sorted(set(self.model.names.values()))

    def update(self, frame, frame_index):
        result = self.model.predict(frame, conf=self.settings.weapon_confidence, verbose=False, **device_kw(self.settings))[0]
        detections = [{"box": tuple(int(v) for v in b.xyxy[0].tolist()), "label": result.names[int(b.cls[0])], "confidence": float(b.conf[0])}
                      for b in result.boxes]
        return [Track(t.id + 1_000_000, t.box, t.label, t.confidence, t.last_frame, t.hits)
                for t in self.tracker.update(detections, frame_index) if t.last_frame == frame_index]


class ByteTrackAdapter:
    """Ultralytics ByteTrack wrapper; model state is reset for each analysis call."""
    def __init__(self, model, settings):
        self.model, self.settings = model, settings
        self.keep = set(settings.bag_classes) | set(settings.alert_object_classes) | {"person"}
        self.class_ids = [int(i) for i, name in model.names.items() if name in self.keep]
        self.fallback = IoUTracker(settings.max_track_gap, settings.match_iou)
        self.use_fallback = False
        self.warned = False

    def update(self, frame, frame_index):
        if self.use_fallback:
            return self._fallback_update(frame, frame_index)
        try:
            result = self.model.track(frame, persist=True, tracker=self.settings.tracker_config,
                                      conf=self.settings.confidence, classes=self.class_ids, verbose=False, **device_kw(self.settings))[0]
        except ModuleNotFoundError as error:
            if error.name != "lap":
                raise
            self.use_fallback = True
            if not self.warned:
                print("WARNING: ByteTrack dependency 'lap' is missing; using the simpler IoU tracker. Install requirements.txt to enable ByteTrack.")
                self.warned = True
            return self._fallback_update(frame, frame_index)
        tracks = []
        if result.boxes is None or result.boxes.id is None:
            return tracks
        ids = result.boxes.id.int().cpu().tolist()
        for box, ident in zip(result.boxes, ids):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            label = result.names[int(box.cls[0])]
            tracks.append(Track(int(ident), (x1, y1, x2, y2), label, float(box.conf[0]), frame_index))
        return tracks

    def _fallback_update(self, frame, frame_index):
        result = self.model.predict(frame, conf=self.settings.confidence, classes=self.class_ids, verbose=False, **device_kw(self.settings))[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append({"box": (x1, y1, x2, y2), "label": result.names[int(box.cls[0])], "confidence": float(box.conf[0])})
        return self.fallback.update(detections, frame_index)


class YoloIoUAdapter:
    def __init__(self, model, settings):
        self.model, self.settings = model, settings
        self.tracker = IoUTracker(settings.max_track_gap, settings.match_iou)
        self.keep = set(settings.bag_classes) | set(settings.alert_object_classes) | {"person"}

    def update(self, frame, frame_index):
        result = self.model.predict(frame, conf=self.settings.confidence, verbose=False, **device_kw(self.settings))[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            label = result.names[int(box.cls[0])]
            if label in self.keep:
                detections.append({"box": (x1, y1, x2, y2), "label": label, "confidence": float(box.conf[0])})
        return self.tracker.update(detections, frame_index)


class SidecarTracker:
    """Replays tracks from JSON: {"frames": {"<frame_index>": [[id, label, x1, y1, x2, y2, conf], ...]}}."""
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        self.frames = raw.get("frames", {})
        self.source = raw.get("source", "sidecar")

    def update(self, frame, frame_index):
        return [Track(int(i), (int(x1), int(y1), int(x2), int(y2)), str(label), float(conf), frame_index)
                for i, label, x1, y1, x2, y2, conf in self.frames.get(str(frame_index), [])]


def frame_quality(frame):
    """Cheap global image-quality cues for the evidence gate (brightness 0-255, Laplacian sharpness)."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (320, max(1, int(320*h/w))), interpolation=cv2.INTER_AREA) if w > 320 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return {"brightness": float(gray.mean()), "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var())}


class SceneMonitor:
    """frame_quality() plus the fraction of the view that differs from a slow-moving reference.

    A person walking past changes part of the view; a covered or moved camera changes most of it.
    """
    def __init__(self, alpha=0.02, pixel_threshold=40):
        self.alpha, self.pixel_threshold, self.reference = alpha, pixel_threshold, None

    def update(self, frame):
        q = frame_quality(frame)
        small = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.reference is None:
            self.reference = small
            q["changed_fraction"] = 0.0
        else:
            q["changed_fraction"] = float((np.abs(small - self.reference) > self.pixel_threshold).mean())
            cv2.accumulateWeighted(small, self.reference, self.alpha)
        return q


def draw_overlay(frame, settings, tracks, alert_count, banner_text="", method="context"):
    for zone in settings.restricted_zones:
        if len(zone.polygon) < 3:
            continue
        points = np.array(zone.polygon, dtype="int32")
        layer = frame.copy()
        cv2.fillPoly(layer, [points], (0, 120, 255))
        cv2.addWeighted(layer, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [points], True, (0, 165, 255), 2)
        cv2.putText(frame, zone.name, tuple(int(v) for v in zone.polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
    for track in tracks:
        x1, y1, x2, y2 = track.box
        color = (40, 200, 40) if track.label == "person" else (0, 0, 255) if track.label in settings.alert_object_classes else (255, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "{} #{} {:.2f}".format(track.label, track.id, track.confidence), (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(frame, "SENTRIX AI | {} | alerts: {}".format(method, alert_count), (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if banner_text:
        cv2.rectangle(frame, (6, 36), (min(frame.shape[1]-6, 12 + 11*len(banner_text[:80])), 66), (20, 20, 200), -1)
        cv2.putText(frame, banner_text[:80], (12, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def analyze(video_path, config_path, live=False, method="context", clip_id=None, scene_id="", session_id="", split="",
            database_path=None, recorded_start=None, detections=None, annotated_out=None, show=False, max_frames=None,
            settings=None, quiet=False):
    if method not in METHODS:
        raise ValueError("method must be one of {}".format(METHODS))
    settings = settings or Settings.from_json(config_path)
    settings.device = resolve_device(settings.device)
    if database_path:
        settings.database = database_path
    source = int(video_path) if isinstance(video_path, str) and video_path.isdigit() else video_path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: {}".format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker, pose_model = None, None
    if method != "motion":
        if detections:
            tracker = SidecarTracker(detections)
        else:
            model = load_yolo(settings.model)
            tracker = ByteTrackAdapter(model, settings) if settings.tracker_backend == "bytetrack" else YoloIoUAdapter(model, settings)
            if settings.pose_enabled:
                pose_model = load_yolo(settings.pose_model)
    engine = MotionBaseline(settings, fps) if method == "motion" else EventEngine(settings, fps, evidence_gate=(method == "context"), context_rules=(method == "context"))
    store = AlertStore(settings.database)
    evidence_dir = os.path.join(settings.evidence_dir, os.path.splitext(os.path.basename(settings.database))[0])
    evidence = EvidenceRecorder(evidence_dir, fps, settings.evidence_pre_seconds, settings.evidence_post_seconds,
                                settings.blur_faces_in_evidence, settings.evidence_max_width)
    recording_id = clip_id or ("camera_{}".format(video_path) if str(video_path).isdigit() else os.path.basename(str(video_path)))
    feature_logger = FeatureLogger(settings.trajectory_csv, recording_id, fps, scene_id, session_id, split) if method != "motion" else None
    writer, cascade = None, None
    if annotated_out:
        os.makedirs(os.path.dirname(annotated_out) or ".", exist_ok=True)
        cascade = load_face_cascade() if settings.blur_faces_in_evidence else None
    frame_index, alert_count, latest_tracks = 0, 0, []
    counts = {}
    scene = SceneMonitor()
    started = time.monotonic()
    if recorded_start:
        recorded_start = datetime.fromisoformat(recorded_start)
        if recorded_start.tzinfo is None:
            recorded_start = recorded_start.replace(tzinfo=ZoneInfo(settings.schedule_timezone))
    banner, banner_until = "", -1.0
    if live and not quiet:
        print("Live analysis started. Press q in the preview window to stop.")
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and frame_index >= max_frames):
            break
        alerts = []
        event_time = time.monotonic()-started if live else frame_index/fps
        wall_clock_time = datetime.now(ZoneInfo(settings.schedule_timezone)) if live else (recorded_start+timedelta(seconds=frame_index/fps) if recorded_start else None)
        if method == "motion":
            alerts = engine.update(frame, frame_index, event_time)
        elif frame_index % max(1, settings.sample_every) == 0:
            latest_tracks = [t for t in tracker.update(frame, frame_index)
                             if (t.box[2]-t.box[0])*(t.box[3]-t.box[1]) >= settings.min_box_area]
            pose = extract_pose_features(pose_model, frame, [t for t in latest_tracks if t.label == "person"], settings.confidence) if pose_model else None
            feature_logger.update(latest_tracks, frame_index, frame.shape, settings.bag_classes, pose, settings.restricted_zones, event_time)
            alerts = engine.update(latest_tracks, frame_index, frame.shape, event_time, wall_clock_time, scene.update(frame), pose)
        for alert in alerts:
            alert["method"] = method
            alert["evidence_privacy"] = "face-region blur applied when evidence is written" if settings.blur_faces_in_evidence else "face blurring disabled"
            alert_id = store.add(recording_id, alert); alert_count += 1
            counts[alert["event_type"]] = counts.get(alert["event_type"], 0) + 1
            evidence.trigger(alert_id)
            if not quiet:
                print(json.dumps(alert, ensure_ascii=False))
            visible_type = alert.get("candidate_event_type", alert["event_type"])
            prefix = "WITHHELD (low evidence): " if alert["event_type"] == "insufficient_evidence" else "ALERT: "
            banner = prefix + visible_type.replace("_", " ") + (" @ {}".format(alert["zone"]) if alert.get("zone") else "")
            banner_until = event_time + 4.0
        person_boxes = [t.box for t in latest_tracks if t.label == "person"]
        for alert_id, path in evidence.add_frame(frame, person_boxes):
            store.set_evidence(alert_id, path)
        if annotated_out or live or show:
            preview = frame.copy()
            if annotated_out and settings.blur_faces_in_evidence:
                blur_faces(preview, person_boxes, cascade)
            draw_overlay(preview, settings, latest_tracks, alert_count, banner if event_time <= banner_until else "", method)
            if annotated_out:
                if writer is None:
                    writer = cv2.VideoWriter(annotated_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (preview.shape[1], preview.shape[0]))
                writer.write(preview)
            if live or show:
                cv2.imshow("SENTRIX AI Monitor", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        frame_index += 1
    cap.release()
    if writer is not None:
        writer.release()
    if feature_logger:
        feature_logger.close()
    for alert_id, path in evidence.close():
        store.set_evidence(alert_id, path)
    store.close()
    if live or show:
        cv2.destroyAllWindows()
    elapsed = max(0.001, time.monotonic()-started)
    summary = {"status": "complete", "source": str(video_path), "clip_id": recording_id, "method": method,
               "detector": "sidecar" if detections else ("none" if method == "motion" else settings.model),
               "frames": frame_index, "duration_seconds": round(frame_index/fps, 2), "elapsed_seconds": round(elapsed, 2),
               "processing_fps": round(frame_index/elapsed, 2), "alerts_written": alert_count, "alerts_by_type": counts,
               "database": settings.database, "annotated_video": annotated_out}
    if not quiet:
        print(json.dumps(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze an authorized recorded video or camera with SENTRIX AI")
    parser.add_argument("video", nargs="?", help="Video path or camera index such as 0")
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--live", action="store_true", help="Live camera mode with preview; press q in its window to stop")
    parser.add_argument("--show", action="store_true", help="Show a preview window while processing a recorded file")
    parser.add_argument("--stream", help="Camera stream URL, such as an authorized RTSP URL")
    parser.add_argument("--method", choices=METHODS, default="context", help="Evaluation configuration")
    parser.add_argument("--detections", help="Replay pre-computed tracks from a JSON sidecar instead of running YOLO")
    parser.add_argument("--annotated-out", help="Write a face-blurred video with zones, tracks and alert banners")
    parser.add_argument("--max-frames", type=int, help="Stop after N frames (quick checks)")
    parser.add_argument("--clip-id", help="Anonymous clip/session ID used in annotations and metrics")
    parser.add_argument("--scene-id", default="", help="Scene/camera identifier for grouping features")
    parser.add_argument("--session-id", default="", help="Recording session identifier for grouping features")
    parser.add_argument("--split", choices=("train", "development", "test"), default="", help="Frozen group split label for feature export")
    parser.add_argument("--database", help="Override SQLite output path; use separate DBs for each method")
    parser.add_argument("--recorded-start", help="ISO timestamp for a recorded clip, required when zone schedules are configured")
    args = parser.parse_args()
    source = args.stream or args.video
    if source is None:
        parser.error("provide a video path, camera index (for example 0), or --stream URL")
    if str(source).isdigit() and not args.live:
        parser.error("camera index requires --live")
    analyze(source, args.config, live=args.live, method=args.method, clip_id=args.clip_id, scene_id=args.scene_id,
            session_id=args.session_id, split=args.split, database_path=args.database, recorded_start=args.recorded_start,
            detections=args.detections, annotated_out=args.annotated_out, show=args.show, max_frames=args.max_frames)

if __name__ == "__main__": main()
