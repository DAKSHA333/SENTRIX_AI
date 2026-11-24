"""Small scene-local IoU tracker; IDs are reset for every input video."""
from dataclasses import dataclass
from typing import Dict, List, Optional


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / float(area_a + area_b - inter) if area_a + area_b > inter else 0.0


@dataclass
class Track:
    id: int
    box: tuple
    label: str
    confidence: float
    last_frame: int
    hits: int = 1


class IoUTracker:
    def __init__(self, max_gap=12, min_iou=0.15):
        self.max_gap, self.min_iou = max_gap, min_iou
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[dict], frame_index: int) -> List[Track]:
        candidates = []
        for track in self.tracks.values():
            if frame_index - track.last_frame <= self.max_gap:
                for index, det in enumerate(detections):
                    if track.label == det["label"]:
                        score = iou(track.box, det["box"])
                        if score >= self.min_iou:
                            candidates.append((score, track.id, index))
        candidates.sort(reverse=True)
        used_tracks, used_dets = set(), set()
        for _, track_id, index in candidates:
            if track_id in used_tracks or index in used_dets:
                continue
            t, d = self.tracks[track_id], detections[index]
            t.box, t.confidence, t.last_frame, t.hits = d["box"], d["confidence"], frame_index, t.hits + 1
            used_tracks.add(track_id); used_dets.add(index)
        for index, det in enumerate(detections):
            if index not in used_dets:
                ident = self.next_id; self.next_id += 1
                self.tracks[ident] = Track(ident, det["box"], det["label"], det["confidence"], frame_index)
        self.tracks = {i: t for i, t in self.tracks.items() if frame_index - t.last_frame <= self.max_gap}
        return list(self.tracks.values())
