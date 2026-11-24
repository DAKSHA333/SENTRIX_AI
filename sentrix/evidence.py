"""Short evidence clips. Frames stay in memory and faces are blurred on write."""
from collections import deque
import os
import time
import cv2


_CASCADE_WARNED = False


def load_face_cascade():
    """Haar face detector, or None (person-box head regions are still blurred) if unavailable."""
    global _CASCADE_WARNED
    path = os.path.join(getattr(getattr(cv2, "data", None), "haarcascades", ""), "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(path) if os.path.isfile(path) else None
    if cascade is None or cascade.empty():
        if not _CASCADE_WARNED:
            print("WARNING: OpenCV Haar face cascade not found; blurring person-box head regions only.")
            _CASCADE_WARNED = True
        return None
    return cascade


def blur_faces(frame, person_boxes=(), cascade=None):
    """Blur conservative head regions of person boxes plus Haar face detections, in place."""
    h, w = frame.shape[:2]
    candidates = []
    for x1, y1, x2, y2 in person_boxes:
        bw, bh = max(1, x2-x1), max(1, y2-y1)
        candidates.append((x1+int(.12*bw), y1+int(.03*bh), x2-int(.12*bw), y1+int(.42*bh)))
    if cascade is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        candidates.extend((x, y, x+fw, y+fh) for x, y, fw, fh in faces)
    for x1, y1, x2, y2 in candidates:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2]
        if roi.size:
            k = max(15, (min(x2-x1, y2-y1)//3) | 1)
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return frame


class EvidenceRecorder:
    def __init__(self, output_dir, fps, pre_seconds=3.0, post_seconds=3.0, blur_faces=True, max_width=960):
        self.output_dir = output_dir
        os.makedirs(output_dir, mode=0o700, exist_ok=True)
        try: os.chmod(output_dir, 0o700)
        except OSError: pass
        self.fps = max(1.0, float(fps))
        self.pre_frames = max(1, int(pre_seconds * self.fps))
        self.post_frames = max(1, int(post_seconds * self.fps))
        self.blur_faces = blur_faces
        self.max_width = max_width or 0
        self.buffer = deque(maxlen=self.pre_frames)
        self.pending = []
        self.face_cascade = load_face_cascade() if blur_faces else None

    def _shrink(self, frame, person_boxes):
        """Downscale buffered frames: a 3 s pre-roll of 1080p frames is ~0.5 GB otherwise."""
        boxes = [tuple(map(int, box)) for box in person_boxes]
        w = frame.shape[1]
        if not self.max_width or w <= self.max_width:
            return frame.copy(), boxes
        scale = self.max_width / float(w)
        small = cv2.resize(frame, (self.max_width, int(round(frame.shape[0]*scale))), interpolation=cv2.INTER_AREA)
        return small, [tuple(int(v*scale) for v in box) for box in boxes]

    def add_frame(self, frame, person_boxes=()):
        item = self._shrink(frame, person_boxes)
        self.buffer.append(item)
        finished = []
        for record in self.pending:
            record["frames"].append(item)
            record["remaining"] -= 1
            if record["remaining"] <= 0:
                record["written"] = self._write(record)
                finished.append(record)
        finished_ids = {id(record) for record in finished}
        self.pending = [record for record in self.pending if id(record) not in finished_ids]
        return [(record["alert_id"], record["path"]) for record in finished if record.get("written")]

    def trigger(self, alert_id):
        record = {"alert_id": alert_id, "frames": list(self.buffer), "remaining": self.post_frames,
                  "path": os.path.join(self.output_dir, "alert-{:06d}-{}.mp4".format(alert_id, time.time_ns()))}
        self.pending.append(record)

    def _blur(self, frame, person_boxes):
        if not self.blur_faces:
            return frame
        return blur_faces(frame, person_boxes, self.face_cascade)

    def _write(self, record):
        if not record["frames"]:
            return False
        first = record["frames"][0][0]
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(record["path"], cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not writer.isOpened():
            return False
        for frame, boxes in record["frames"]:
            writer.write(self._blur(frame.copy(), boxes))
        writer.release()
        ok = os.path.isfile(record["path"]) and os.path.getsize(record["path"]) > 0
        if ok:
            try: os.chmod(record["path"], 0o600)
            except OSError: pass
        return ok

    def close(self):
        for record in self.pending:
            record["written"] = self._write(record)
        result = [(record["alert_id"], record["path"]) for record in self.pending if record.get("written")]
        self.pending.clear()
        return result
