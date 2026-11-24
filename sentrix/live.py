"""SENTRIX AI live monitor: camera -> detection -> alerts -> website + notifications.

    python -m sentrix.live --camera 0 --config config.live.json --notify notify.json
    # then open http://127.0.0.1:8080

Also runs on a recorded clip (--video FILE [--loop]) and on pre-computed tracks
(--detections FILE.json) for testing without a model. Standard library web server:
  /                    website (sentrix/web/index.html)
  /video               live annotated MJPEG stream
  /api/stream          server-sent events: new alerts and notification results
  /api/status          camera/model/fps/channel status
  /api/alerts          recent alerts (JSON)
  /api/alerts/<id>/decision   POST {"decision": "confirmed|dismissed|escalated", "note": ""}
  /api/test-alert      POST: send a test notification
  /snapshots/<file>    face-blurred alert snapshots
Bind to 127.0.0.1 (default). Set SENTRIX_WEB_PASSWORD to require a password.
"""
import argparse
import base64
import hmac
import json
import os
import queue
import re
import sqlite3
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import cv2
from .analyze import (SceneMonitor, SidecarTracker, ByteTrackAdapter, YoloIoUAdapter, WeaponDetector,
                      draw_overlay, load_yolo, resolve_device)
from .config import Settings
from .events import EventEngine
from .evidence import EvidenceRecorder, blur_faces, load_face_cascade
from .notify import Notifier, title_for
from .pose import extract_pose_features
from .store import AlertStore

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class LiveMonitor(threading.Thread):
    def __init__(self, source, settings, notifier, camera_name="Camera 1", detections=None, loop=False,
                 mirror=False, snapshot_dir="data/live_snapshots", jpeg_quality=75, capture=None):
        super().__init__(daemon=True, name="sentrix-monitor")
        self.capture = capture
        self.source, self.settings, self.notifier = source, settings, notifier
        self.camera_name, self.detections, self.loop, self.mirror = camera_name, detections, loop, mirror
        self.snapshot_dir, self.jpeg_quality = snapshot_dir, jpeg_quality
        os.makedirs(snapshot_dir, mode=0o700, exist_ok=True)
        self.frame_lock = threading.Condition()
        self.latest_jpeg, self.frame_seq = None, 0
        self.subscribers, self.sub_lock = [], threading.Lock()
        self.stop_event = threading.Event()
        self.state = {"status": "starting", "error": None, "fps": 0.0, "frames": 0, "alerts": 0, "people": 0,
                      "objects": [], "started": time.strftime("%Y-%m-%d %H:%M:%S"), "detector": "", "resolution": None}

    # ------------------------------------------------------------------ pub/sub
    def subscribe(self):
        q = queue.Queue(maxsize=200)
        with self.sub_lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.sub_lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def publish(self, kind, data):
        with self.sub_lock:
            for q in list(self.subscribers):
                try:
                    q.put_nowait((kind, data))
                except queue.Full:
                    pass

    def stop(self):
        self.stop_event.set()

    # ------------------------------------------------------------------ setup
    def _open(self):
        if self.capture is not None and self.capture.isOpened():
            return self.capture
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera/video '{}'. On macOS allow camera access for Terminal "
                               "(System Settings > Privacy & Security > Camera) and close other apps using the camera.".format(self.source))
        return cap

    def _build_detectors(self):
        s = self.settings
        if self.detections:
            self.state["detector"] = "replay: {}".format(os.path.basename(self.detections))
            return SidecarTracker(self.detections), None, None
        model = load_yolo(s.model)
        tracker = ByteTrackAdapter(model, s) if s.tracker_backend == "bytetrack" else YoloIoUAdapter(model, s)
        weapon = None
        if s.weapon_model:
            weapon = WeaponDetector(s)
            s.alert_object_classes = tuple(sorted(set(s.alert_object_classes) | set(weapon.labels)))
        pose = load_yolo(s.pose_model) if (s.pose_enabled or s.enable_arms_raised) else None
        self.state["detector"] = "{}{}{} on {}".format(s.model, " + " + os.path.basename(s.weapon_model) if weapon else "",
                                                       " + pose" if pose else "", s.device or "default device")
        return tracker, weapon, pose

    # ------------------------------------------------------------------ main loop
    def run(self):
        try:
            self._run()
        except Exception as exc:
            self.state.update(status="error", error="{}: {}".format(type(exc).__name__, exc))
            self.publish("status", self.state)
            print("SENTRIX monitor stopped:", self.state["error"])

    def _run(self):
        s = self.settings
        cap = self._open()
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        is_file = not str(self.source).isdigit() and os.path.isfile(str(self.source))
        tracker, weapon, pose_model = self._build_detectors()
        engine = EventEngine(s, fps_src, evidence_gate=True, context_rules=True)
        store = AlertStore(s.database)
        evidence = EvidenceRecorder(os.path.join(s.evidence_dir, "live"), fps_src, s.evidence_pre_seconds,
                                    s.evidence_post_seconds, s.blur_faces_in_evidence, s.evidence_max_width)
        cascade = load_face_cascade() if s.blur_faces_in_evidence else None
        scene = SceneMonitor()
        self.state["status"] = "live"
        started, frame_index, tick, tick_frames = time.monotonic(), 0, time.monotonic(), 0
        tracks, banner, banner_until, failures = [], "", -1.0, 0
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                if is_file and self.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); engine = EventEngine(s, fps_src, True, True); scene = SceneMonitor()
                    frame_index = 0; started = time.monotonic(); continue
                failures += 1
                if is_file or failures > 50:
                    break
                time.sleep(0.05); continue
            failures = 0
            if self.mirror:
                frame = cv2.flip(frame, 1)
            now = frame_index / fps_src if is_file else time.monotonic() - started
            if self.state["resolution"] is None:
                self.state["resolution"] = "{}x{}".format(frame.shape[1], frame.shape[0])
            alerts = []
            if frame_index % max(1, s.sample_every) == 0:
                tracks = [t for t in tracker.update(frame, frame_index) if (t.box[2]-t.box[0])*(t.box[3]-t.box[1]) >= s.min_box_area
                          or t.label in s.alert_object_classes]
                if weapon:
                    tracks += weapon.update(frame, frame_index)
                people = [t for t in tracks if t.label == "person"]
                pose = extract_pose_features(pose_model, frame, people, s.confidence) if pose_model and people else None
                alerts = engine.update(tracks, frame_index, frame.shape, now, datetime.now().astimezone(), scene.update(frame), pose)
                self.state["people"] = len(people)
                self.state["objects"] = sorted({t.label for t in tracks if t.label != "person"})
            person_boxes = [t.box for t in tracks if t.label == "person"]
            if alerts:
                blurred = blur_faces(frame.copy(), person_boxes, cascade) if s.blur_faces_in_evidence else frame.copy()
            for alert in alerts:
                alert["method"] = "context"
                alert["camera"] = self.camera_name
                alert["wall_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                alert["title"] = title_for(alert)
                snap_name = "alert-{}-{}.jpg".format(time.strftime("%Y%m%d-%H%M%S"), self.state["alerts"] + 1)
                snap = draw_overlay(blurred.copy(), s, tracks, self.state["alerts"] + 1, alert["title"], "live")
                self._highlight(snap, alert)
                cv2.imwrite(os.path.join(self.snapshot_dir, snap_name), snap, [cv2.IMWRITE_JPEG_QUALITY, 85])
                alert["snapshot"] = snap_name
                alert_id = store.add(self.camera_name, alert)
                evidence.trigger(alert_id)
                self.state["alerts"] += 1
                notified = self.notifier.notify(dict(alert, id=alert_id), os.path.join(self.snapshot_dir, snap_name))
                record = {"id": alert_id, "decision": "unreviewed", "notified": notified, **alert}
                self.publish("alert", record)
                print("[{}] {} - {}".format(alert["wall_time"], alert["title"], alert["reason"]))
                prefix = "NEEDS REVIEW: " if alert["event_type"] == "insufficient_evidence" else "ALERT: "
                banner, banner_until = prefix + alert["title"], now + 5.0
            for alert_id, path in evidence.add_frame(frame, person_boxes):
                store.set_evidence(alert_id, path)
            view = draw_overlay(frame.copy(), s, tracks, self.state["alerts"], banner if now <= banner_until else "", "live")
            ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                with self.frame_lock:
                    self.latest_jpeg, self.frame_seq = buf.tobytes(), self.frame_seq + 1
                    self.frame_lock.notify_all()
            frame_index += 1; tick_frames += 1
            self.state["frames"] = frame_index
            if time.monotonic() - tick >= 1.0:
                self.state["fps"] = round(tick_frames / (time.monotonic() - tick), 1); tick, tick_frames = time.monotonic(), 0
            if is_file and self.detections:
                time.sleep(0.5 / fps_src)       # replay runs at ~2x real time so the website can follow
        cap.release()
        for alert_id, path in evidence.close():
            store.set_evidence(alert_id, path)
        store.close()
        self.state["status"] = "stopped" if self.state["status"] != "error" else "error"
        self.publish("status", self.state)

    @staticmethod
    def _highlight(img, alert):
        box = (alert.get("measurements") or {}).get("bounding_box_px")
        if box:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), 4)


# ---------------------------------------------------------------------------- web server
class Handler(BaseHTTPRequestHandler):
    server_version = "SENTRIX/1.0"

    def log_message(self, fmt, *args):
        pass

    # helpers
    def _authorized(self):
        password = self.server.password
        if not password:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Basic "):
            try:
                _, _, given = base64.b64decode(header[6:]).decode().partition(":")
                if hmac.compare_digest(given, password):
                    return True
            except Exception:
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="SENTRIX AI"')
        self.end_headers()
        return False

    def _json(self, data, status=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        if not os.path.isfile(path):
            return self._json({"error": "not found"}, 404)
        with open(path, "rb") as handle:
            body = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # routes
    def do_GET(self):
        if not self._authorized():
            return
        url = urlparse(self.path)
        path = url.path
        if path in ("/", "/index.html"):
            return self._file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
        if path == "/favicon.ico":
            body = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#ff4d4f"/><circle cx="16" cy="16" r="6" fill="#0b0f14"/></svg>'
            self.send_response(200); self.send_header("Content-Type", "image/svg+xml"); self.send_header("Content-Length", str(len(body))); self.end_headers()
            return self.wfile.write(body)
        if path == "/video":
            return self._mjpeg()
        if path == "/api/stream":
            return self._sse()
        if path == "/api/status":
            mon = self.server.monitor
            return self._json({**mon.state, "camera": mon.camera_name, "channels": self.server.notifier.channels(),
                               "notification_log": list(self.server.notifier.log)[:20],
                               "watching": watch_list(mon.settings)})
        if path == "/api/alerts":
            limit = max(1, min(500, int(parse_qs(url.query).get("limit", ["100"])[0])))
            return self._json(list_alerts(self.server.db_path, limit))
        if path.startswith("/snapshots/"):
            name = path.rsplit("/", 1)[-1]
            if not SAFE_NAME.match(name):
                return self._json({"error": "bad name"}, 400)
            return self._file(os.path.join(self.server.monitor.snapshot_dir, name), "image/jpeg")
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._authorized():
            return
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except json.JSONDecodeError:
            return self._json({"error": "invalid JSON"}, 400)
        m = re.match(r"^/api/alerts/(\d+)/decision$", path)
        if m:
            decision = body.get("decision")
            if decision not in ("confirmed", "dismissed", "escalated"):
                return self._json({"error": "decision must be confirmed, dismissed or escalated"}, 400)
            store = AlertStore(self.server.db_path)
            try:
                store.decide(int(m.group(1)), decision, actor="web_reviewer", note=str(body.get("note", ""))[:500])
            finally:
                store.close()
            self.server.monitor.publish("decision", {"id": int(m.group(1)), "decision": decision})
            if decision == "escalated":
                rows = [a for a in list_alerts(self.server.db_path, 500) if a["id"] == int(m.group(1))]
                if rows:
                    a = dict(rows[0]["payload"], event_type=rows[0]["payload"].get("event_type"))
                    a["reason"] = "ESCALATED by reviewer. " + a.get("reason", "")
                    snap = os.path.join(self.server.monitor.snapshot_dir, a.get("snapshot", "")) if a.get("snapshot") else None
                    self.server.notifier.notify(a, snap, force=True)
            return self._json({"id": int(m.group(1)), "decision": decision})
        if path == "/api/test-alert":
            sent = self.server.notifier.test()
            return self._json({"queued": sent, "channels": self.server.notifier.channels()})
        return self._json({"error": "not found"}, 404)

    def _mjpeg(self):
        mon = self.server.monitor
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seq = -1
        try:
            while not self.server.stopping:
                with mon.frame_lock:
                    mon.frame_lock.wait_for(lambda: mon.frame_seq != seq or self.server.stopping, timeout=2.0)
                    jpeg, seq = mon.latest_jpeg, mon.frame_seq
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                if self.server.max_stream_frames and seq >= self.server.max_stream_frames:
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse(self):
        mon = self.server.monitor
        q = mon.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(b"event: hello\ndata: {}\n\n"); self.wfile.flush()
            while not self.server.stopping:
                try:
                    kind, data = q.get(timeout=15)
                    self.wfile.write("event: {}\ndata: {}\n\n".format(kind, json.dumps(data, default=str)).encode())
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            mon.unsubscribe(q)


def list_alerts(db_path, limit=100):
    if not os.path.isfile(db_path):
        return []
    db = sqlite3.connect(db_path, timeout=10)
    try:
        rows = db.execute("SELECT id, video, event_type, timestamp_seconds, payload, decision, created_at, evidence_path, reviewer_note "
                          "FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        db.close()
    out = []
    for r in rows:
        payload = json.loads(r[4])
        out.append({"id": r[0], "camera": r[1], "event_type": r[2], "decision": r[5], "created_at": r[6],
                    "reviewer_note": r[8], "title": payload.get("title") or title_for(payload), "payload": payload, **{
                    k: payload.get(k) for k in ("reason", "snapshot", "wall_time", "evidence_quality", "evidence_reasons", "measurements", "candidate_event_type", "zone")}})
    return out


def watch_list(s):
    items = [("Objects", ", ".join(s.alert_object_classes) or "none")]
    items.append(("Restricted zones", ", ".join(z.name for z in s.restricted_zones if z.alert_on_entry) or "none configured"))
    if any(z.enable_dwell for z in s.restricted_zones) and s.enable_dwell_pacing:
        items.append(("Loitering", "over {:.0f}s in a dwell zone".format(s.dwell_seconds)))
    items.append(("Unattended bag", "{} after {:.0f}s".format(", ".join(s.bag_classes), s.unattended_seconds)))
    if s.enable_tamper: items.append(("Camera tampering", "lens covered / blocked / blacked out{} for {:.0f}s".format(" / moved" if s.tamper_detect_moved else "", s.tamper_sustain_seconds)))
    if s.enable_running: items.append(("Rapid movement", "> {} body-heights/s".format(s.running_body_heights_per_s)))
    if s.enable_crowd: items.append(("Crowding", ">= {} people".format(s.crowd_alert_count)))
    if s.enable_arms_raised: items.append(("Both arms raised", "held {:.0f}s (pose model)".format(s.arms_raised_sustain_seconds)))
    if s.enable_dwell_pacing: items.append(("Pacing", "{} reversals in {:.0f}s".format(s.pacing_reversals, s.pacing_window_seconds)))
    if s.enable_counterflow: items.append(("Counter-flow", "needs a moving crowd"))
    if s.enable_dispersion: items.append(("Group dispersal", "needs >= {} people".format(s.dispersion_min_people)))
    return [{"name": a, "detail": b} for a, b in items]


def open_camera(index, attempts=3):
    for attempt in range(attempts):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
        time.sleep(1.0)
    others = []
    for alt in range(4):
        if alt == index:
            continue
        test = cv2.VideoCapture(alt)
        if test.isOpened():
            others.append(str(alt))
        test.release()
    hint = " Cameras that did open: {} (try --camera {}).".format(", ".join(others), others[0]) if others else ""
    raise SystemExit("Could not open camera {}. Quit Terminal (Cmd+Q), reopen, check System Settings > Privacy & Security > Camera, close FaceTime/Zoom/Meet.{}".format(index, hint))


def serve(monitor, notifier, db_path, host="127.0.0.1", port=8080, password=""):
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.monitor, server.notifier, server.db_path, server.password = monitor, notifier, db_path, password
    server.stopping, server.max_stream_frames = False, 0
    return server


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", default="0", help="Camera index (0 = built-in/front camera) or authorized stream URL")
    src.add_argument("--video", help="Recorded clip instead of a camera")
    p.add_argument("--loop", action="store_true", help="Loop --video")
    p.add_argument("--detections", help="Replay pre-computed tracks (no model needed)")
    p.add_argument("--config", default="config.live.json")
    p.add_argument("--notify", default="notify.json", help="Notification contacts/channels file")
    p.add_argument("--camera-name", default="Front camera (demo)")
    p.add_argument("--mirror", action="store_true", help="Mirror the image like a selfie view")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8080)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    settings = Settings.from_json(a.config)
    settings.device = resolve_device(settings.device)
    notifier = Notifier.from_file(a.notify, a.camera_name)
    if not notifier.cfg.get("dashboard_url"):
        notifier.dashboard_url = "http://{}:{}".format(a.host, a.port)
    capture = None
    source = a.video or a.camera
    if not a.video and str(source).isdigit():
        capture = open_camera(int(source))
    monitor = LiveMonitor(source, settings, notifier, a.camera_name, a.detections, a.loop, a.mirror, capture=capture)
    server = serve(monitor, notifier, settings.database, a.host, a.port, os.environ.get("SENTRIX_WEB_PASSWORD", ""))
    if a.host not in ("127.0.0.1", "localhost") and not server.password:
        print("WARNING: serving beyond localhost without SENTRIX_WEB_PASSWORD. Anyone on the network can view the camera.")
    monitor.start()
    url = "http://{}:{}".format("127.0.0.1" if a.host == "0.0.0.0" else a.host, a.port)
    print("SENTRIX AI live monitor at {}  (Ctrl+C to stop)".format(url))
    print("Notification channels:", ", ".join(notifier.channels()) or "none - website alerts only (see notify.example.json)")
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        server.stopping = True
        monitor.stop()
        monitor.join(timeout=5)
        server.server_close()

if __name__ == "__main__": main()
