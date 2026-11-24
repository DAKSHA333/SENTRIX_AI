"""Alert notifications to the responsible people: Telegram, email (SMTP) and webhooks.

Standard library only. Configure in notify.json (copy notify.example.json; keep it out of git):

  {"telegram": {"bot_token": "...", "chat_ids": ["123456789"]},
   "email": {"smtp_host": "smtp.gmail.com", "smtp_port": 587, "username": "you@gmail.com",
             "password": "<app password>", "from": "you@gmail.com", "to": ["guard@example.com"]},
   "webhooks": ["https://..."],
   "cooldown_seconds": 30, "notify_withheld": false, "dashboard_url": "http://127.0.0.1:8080"}

Messages describe what the detector saw and ask a person to review it. They never
name or characterise anyone.

  python -m sentrix.notify --config notify.json --test     # send a test message to every channel
"""
import argparse
import json
import mimetypes
import os
import queue
import smtplib
import ssl
import threading
import time
import urllib.request
import uuid
from collections import deque
from email.message import EmailMessage

TITLES = {
    "potentially_hazardous_object_detection": "Possible {label} detected",
    "restricted_zone_entry": "Restricted zone entry",
    "potentially_unattended_object": "Possibly unattended bag",
    "prolonged_zone_dwell": "Prolonged stay in zone",
    "repeated_path_reversals": "Repeated back-and-forth movement",
    "sustained_counter_flow": "Movement against crowd flow",
    "sudden_group_dispersion": "Sudden group dispersal",
    "camera_tampering_suspected": "Camera view obstructed or moved",
    "crowding_threshold_exceeded": "Crowding limit exceeded",
    "rapid_movement": "Rapid movement",
    "both_arms_raised": "Both arms raised",
    "insufficient_evidence": "Needs review (low evidence): {candidate}",
    "test": "Test alert",
}


def title_for(alert):
    label = (alert.get("measurements") or {}).get("object_label", "object")
    candidate = (alert.get("candidate_event_type") or "").replace("_", " ")
    return TITLES.get(alert.get("event_type"), alert.get("event_type", "alert").replace("_", " ").capitalize()).format(label=label, candidate=candidate)


def format_message(alert, camera="Camera", dashboard_url=""):
    m = alert.get("measurements") or {}
    lines = ["SENTRIX AI alert: {}".format(title_for(alert)),
             "Camera: {}".format(camera),
             "Time: {}".format(alert.get("wall_time") or time.strftime("%Y-%m-%d %H:%M:%S"))]
    if m.get("detector_confidence") is not None:
        lines.append("Detector confidence: {:.0%}".format(float(m["detector_confidence"])))
    if alert.get("zone"):
        lines.append("Zone: {}".format(alert["zone"]))
    lines.append("What was observed: {}".format(alert.get("reason", "")))
    if alert.get("evidence_reasons"):
        lines.append("Evidence flags: {}".format(", ".join(r.lower().replace("_", " ") for r in alert["evidence_reasons"])))
    lines.append("Automated detection - please review the image and confirm or dismiss before acting.")
    if dashboard_url:
        lines.append("Review: {}".format(dashboard_url))
    return "\n".join(lines)


def _multipart(fields, files):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += "--{}\r\nContent-Disposition: form-data; name=\"{}\"\r\n\r\n{}\r\n".format(boundary, name, value).encode()
    for name, path in files.items():
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as handle:
            data = handle.read()
        body += "--{}\r\nContent-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\nContent-Type: {}\r\n\r\n".format(
            boundary, name, os.path.basename(path), mime).encode() + data + b"\r\n"
    body += "--{}--\r\n".format(boundary).encode()
    return bytes(body), "multipart/form-data; boundary={}".format(boundary)


class Notifier:
    def __init__(self, config=None, camera="Camera 1", transport=None):
        self.cfg = config or {}
        self.camera = camera
        self.cooldown = float(self.cfg.get("cooldown_seconds", 30))
        self.notify_withheld = bool(self.cfg.get("notify_withheld", False))
        self.dashboard_url = self.cfg.get("dashboard_url", "")
        self.log = deque(maxlen=100)
        self.last_sent = {}
        self._transport = transport or self._http
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True, name="sentrix-notifier").start()

    @classmethod
    def from_file(cls, path, camera="Camera 1"):
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                return cls(json.load(handle), camera)
        return cls({}, camera)

    # ------------------------------------------------------------------ public
    def channels(self):
        out = []
        tg = self.cfg.get("telegram") or {}
        if tg.get("bot_token") and tg.get("chat_ids"):
            out.append("telegram")
        em = self.cfg.get("email") or {}
        if em.get("smtp_host") and em.get("to"):
            out.append("email")
        if self.cfg.get("webhooks"):
            out.append("webhook")
        return out

    def notify(self, alert, image_path=None, force=False):
        """Queue a notification. Returns False when skipped (no channels, withheld, cooldown)."""
        if not self.channels():
            return False
        if alert.get("event_type") == "insufficient_evidence" and not (self.notify_withheld or force):
            return False
        key = (alert.get("event_type"), (alert.get("measurements") or {}).get("object_label"), alert.get("zone"))
        now = time.monotonic()
        with self._lock:
            if not force and now - self.last_sent.get(key, -1e9) < self.cooldown:
                self._record("all", False, "skipped: cooldown for {}".format(title_for(alert)))
                return False
            self.last_sent[key] = now
        self._queue.put((alert, image_path))
        return True

    def test(self, image_path=None):
        alert = {"event_type": "test", "reason": "This is a test message from the SENTRIX AI live monitor.", "measurements": {}}
        return self.notify(alert, image_path, force=True)

    def flush(self, timeout=10.0):
        end = time.monotonic() + timeout
        while (not self._queue.empty() or getattr(self, "_busy", False)) and time.monotonic() < end:
            time.sleep(0.05)

    # ------------------------------------------------------------------ worker
    def _record(self, channel, ok, detail):
        self.log.appendleft({"time": time.strftime("%H:%M:%S"), "channel": channel, "ok": ok, "detail": str(detail)[:200]})

    def _worker(self):
        while True:
            alert, image = self._queue.get()
            self._busy = True
            text = format_message(alert, self.camera, self.dashboard_url)
            for name, fn in (("telegram", self._telegram), ("email", self._email), ("webhook", self._webhook)):
                if name not in self.channels():
                    continue
                try:
                    detail = fn(alert, text, image if image and os.path.isfile(image) else None)
                    self._record(name, True, detail or "sent")
                except Exception as exc:          # network/credential errors must never stop monitoring
                    self._record(name, False, "{}: {}".format(type(exc).__name__, exc))
            self._busy = False

    def _http(self, url, data, content_type, timeout=15):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": content_type}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(500)

    def _telegram(self, alert, text, image):
        tg = self.cfg["telegram"]
        base = "https://api.telegram.org/bot{}/".format(tg["bot_token"])
        sent = 0
        for chat in tg["chat_ids"]:
            if image:
                body, ctype = _multipart({"chat_id": chat, "caption": text[:1000]}, {"photo": image})
                status, _ = self._transport(base + "sendPhoto", body, ctype)
            else:
                body = json.dumps({"chat_id": chat, "text": text}).encode()
                status, _ = self._transport(base + "sendMessage", body, "application/json")
            if status >= 300:
                raise RuntimeError("Telegram HTTP {}".format(status))
            sent += 1
        return "sent to {} chat(s)".format(sent)

    def _email(self, alert, text, image):
        em = self.cfg["email"]
        msg = EmailMessage()
        msg["Subject"] = "[SENTRIX AI] {} - {}".format(title_for(alert), self.camera)
        msg["From"] = em.get("from") or em.get("username")
        msg["To"] = ", ".join(em["to"])
        msg.set_content(text)
        if image:
            with open(image, "rb") as handle:
                msg.add_attachment(handle.read(), maintype="image", subtype="jpeg", filename=os.path.basename(image))
        port = int(em.get("smtp_port", 587))
        if port == 465:
            server = smtplib.SMTP_SSL(em["smtp_host"], port, context=ssl.create_default_context(), timeout=20)
        else:
            server = smtplib.SMTP(em["smtp_host"], port, timeout=20)
            server.starttls(context=ssl.create_default_context())
        with server:
            if em.get("username"):
                server.login(em["username"], em["password"])
            server.send_message(msg)
        return "sent to {}".format(", ".join(em["to"]))

    def _webhook(self, alert, text, image):
        payload = json.dumps({"title": title_for(alert), "text": text, "camera": self.camera, "alert": alert}, default=str).encode()
        for url in self.cfg["webhooks"]:
            status, _ = self._transport(url, payload, "application/json")
            if status >= 300:
                raise RuntimeError("webhook HTTP {}".format(status))
        return "posted to {} webhook(s)".format(len(self.cfg["webhooks"]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="notify.json")
    p.add_argument("--test", action="store_true", help="Send a test message to every configured channel")
    a = p.parse_args()
    n = Notifier.from_file(a.config)
    print("Configured channels:", n.channels() or "none (copy notify.example.json to notify.json and fill it in)")
    if a.test and n.channels():
        n.test(); time.sleep(0.2); n.flush(30)
        for row in reversed(n.log):
            print("{channel:<9} {status} {detail}".format(status="OK  " if row["ok"] else "FAIL", **row))

if __name__ == "__main__": main()
