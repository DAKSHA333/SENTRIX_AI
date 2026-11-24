import copy
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
import cv2
import numpy as np
from sentrix.config import Settings
from sentrix.live import LiveMonitor, serve
from sentrix.notify import Notifier, format_message


def make_clip(root, frames=60, fps=10):
    """Blank clip + sidecar: a person throughout, a knife visible from frame 10."""
    video = os.path.join(root, "clip.mp4")
    w = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    sidecar = {"frames": {}}
    for f in range(frames):
        img = np.full((240, 320, 3), 120, np.uint8)
        w.write(img)
        dets = [[1, "person", 100, 40, 160, 230, 0.9]]
        if f >= 10:
            dets.append([2, "knife", 170, 120, 210, 150, 0.8])
        sidecar["frames"][str(f)] = dets
    w.release()
    path = os.path.join(root, "clip.tracks.json")
    with open(path, "w") as handle:
        json.dump(sidecar, handle)
    return video, path


class FakeTransport:
    def __init__(self):
        self.calls = []

    def __call__(self, url, data, content_type, timeout=15):
        self.calls.append((url, content_type, len(data)))
        return 200, b"{}"


class NotifierTests(unittest.TestCase):
    def test_telegram_photo_and_cooldown(self):
        transport = FakeTransport()
        n = Notifier({"telegram": {"bot_token": "T", "chat_ids": ["1", "2"]}, "cooldown_seconds": 60}, transport=transport)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8fakejpeg")
        alert = {"event_type": "potentially_hazardous_object_detection", "reason": "r", "measurements": {"object_label": "knife", "detector_confidence": .8}}
        self.assertTrue(n.notify(alert, f.name))
        self.assertFalse(n.notify(alert, f.name))          # cooldown
        self.assertFalse(n.notify({"event_type": "insufficient_evidence", "reason": "x"}))   # withheld not pushed
        time.sleep(0.1); n.flush()
        self.assertEqual(len(transport.calls), 2)
        self.assertTrue(transport.calls[0][0].endswith("/sendPhoto"))
        self.assertIn("multipart/form-data", transport.calls[0][1])
        os.unlink(f.name)

    def test_message_names_object_not_people(self):
        text = format_message({"event_type": "potentially_hazardous_object_detection", "reason": "seen",
                               "measurements": {"object_label": "knife", "detector_confidence": .62}}, "Front camera")
        self.assertIn("Possible knife detected", text)
        self.assertIn("62%", text)
        self.assertIn("review", text.lower())

    def test_no_channels_means_no_send(self):
        self.assertFalse(Notifier({}).notify({"event_type": "rapid_movement", "reason": "x"}))


class LiveServerTests(unittest.TestCase):
    def test_camera_to_website_flow(self):
        with tempfile.TemporaryDirectory() as root:
            video, sidecar = make_clip(root)
            s = Settings(alert_object_classes=("knife",), object_confirm_samples=3, low_quality_min_object_height=5,
                         low_quality_min_person_height=20, sample_every=1, min_box_area=50, enable_tamper=True,
                         database=os.path.join(root, "live.db"), evidence_dir=os.path.join(root, "ev"),
                         evidence_pre_seconds=0.5, evidence_post_seconds=0.5)
            transport = FakeTransport()
            notifier = Notifier({"webhooks": ["http://example.invalid/hook"]}, "Test cam", transport=transport)
            mon = LiveMonitor(video, s, notifier, "Test cam", detections=sidecar, snapshot_dir=os.path.join(root, "snaps"))
            server = serve(mon, notifier, s.database, "127.0.0.1", 0)
            port = server.server_address[1]
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = "http://127.0.0.1:{}".format(port)
            events = []

            def listen():
                with urllib.request.urlopen(base + "/api/stream", timeout=10) as resp:
                    for raw in resp:
                        line = raw.decode().strip()
                        if line.startswith("event:"):
                            events.append(line.split(":", 1)[1].strip())
                        if "alert" in events:
                            return
            listener = threading.Thread(target=listen, daemon=True); listener.start()
            time.sleep(0.3)
            mon.start()
            mon.join(timeout=20)
            listener.join(timeout=5)
            try:
                self.assertIn("alert", events)
                html = urllib.request.urlopen(base + "/").read().decode()
                self.assertIn("SENTRIX AI", html)
                status = json.loads(urllib.request.urlopen(base + "/api/status").read())
                self.assertEqual(status["camera"], "Test cam")
                self.assertIn("webhook", status["channels"])
                alerts = json.loads(urllib.request.urlopen(base + "/api/alerts").read())
                knife = [a for a in alerts if (a["measurements"] or {}).get("object_label") == "knife"]
                self.assertEqual(len(knife), 1)
                self.assertEqual(knife[0]["title"], "Possible knife detected")
                snap = urllib.request.urlopen(base + "/snapshots/" + knife[0]["snapshot"]).read()
                self.assertTrue(snap.startswith(b"\xff\xd8"))
                req = urllib.request.Request(base + "/api/alerts/{}/decision".format(knife[0]["id"]),
                                             data=json.dumps({"decision": "confirmed"}).encode(), headers={"Content-Type": "application/json"})
                self.assertEqual(json.loads(urllib.request.urlopen(req).read())["decision"], "confirmed")
                alerts = json.loads(urllib.request.urlopen(base + "/api/alerts").read())
                self.assertEqual([a["decision"] for a in alerts if a["id"] == knife[0]["id"]], ["confirmed"])
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(base + "/snapshots/..%2Fsecret")
                stream = urllib.request.urlopen(base + "/video", timeout=5)
                self.assertTrue(stream.read(20).startswith(b"--frame"))
                stream.close()
                notifier.flush()
                self.assertTrue(any("example.invalid" in c[0] for c in transport.calls))
            finally:
                server.stopping = True
                server.shutdown(); server.server_close()


if __name__ == "__main__": unittest.main()
