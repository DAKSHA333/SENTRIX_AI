"""Observable, configurable event logic with an explicit evidence gate.

Two rule sets share this engine so the proposal's comparison is like-for-like:

* context (Proposed C): zone hysteresis, person-separation context for bags,
  schedules, optional extensions, and an evidence-quality gate that abstains with
  INSUFFICIENT_EVIDENCE plus machine-readable reasons.
* single_rule (Baseline B): one fixed rule per event (first boundary crossing;
  bag stationary for T seconds), no scene context and no abstention.
"""
from collections import deque
from math import hypot
from datetime import time as clock_time
from zoneinfo import ZoneInfo
import cv2
import numpy as np


def center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def foot_point(box):
    """Bottom-centre of a box: where a person stands on the ground plane."""
    return ((box[0] + box[2]) // 2, box[3])


def point_in_zone(point, zone):
    return cv2.pointPolygonTest(np.array(zone.polygon, dtype="int32"), (float(point[0]), float(point[1])), False) >= 0


def vector(a, b):
    return (b[0] - a[0], b[1] - a[1])


def magnitude(v):
    return hypot(v[0], v[1])


def cosine(a, b):
    den = magnitude(a) * magnitude(b)
    return (a[0] * b[0] + a[1] * b[1]) / den if den else 0.0


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


class EventEngine:
    def __init__(self, settings, fps, evidence_gate=True, context_rules=None):
        self.settings, self.fps = settings, max(1.0, fps)
        self.evidence_gate = evidence_gate
        self.context = evidence_gate if context_rules is None else context_rules
        self.confirm_samples = max(1, int(settings.zone_entry_confirm_samples)) if self.context else 1
        self.anchor = foot_point if settings.zone_anchor == "bottom" else center
        self.previous_zone_state = {}
        self.inside_streak = {}
        self.zone_entered_at = {}
        self.zone_seen_outside = set()
        self.zone_last_alerted = {}
        self.objects = {}
        self.alerted_objects = set()
        self.alerted_dwell = set()
        self.alerted_pacing = set()
        self.alerted_counterflow = set()
        self.dispersion_since = None
        self.dispersion_alerted = False
        self.history = {}
        self.counterflow_since = {}
        self.last_seen = {}
        self.update_count = 0
        self.frame_quality = None
        self.object_hits = {}
        self.object_label_last_alert = {}
        self.tamper_since = None
        self.tamper_clear_since = None
        self.tamper_alerted = False
        self.crowd_since = None
        self.crowd_alerted = False
        self.running_since = {}
        self.alerted_running = set()
        self.arms_since = {}
        self.alerted_arms = set()

    # ------------------------------------------------------------------ evidence
    def _alert(self, event_type, reason_code, reason, now, track=None, zone=None, quality="SUFFICIENT", measurements=None, reasons=None):
        reasons = list(reasons or [])
        if self.evidence_gate and quality != "SUFFICIENT":
            return {"event_type": "insufficient_evidence", "candidate_event_type": event_type,
                    "timestamp_seconds": round(now, 2), "zone": zone, "track_id": track.id if track else None,
                    "evidence_quality": "INSUFFICIENT_EVIDENCE", "reason_code": "EVIDENCE_GATE_WITHHELD",
                    "reason": "A candidate {} did not meet the evidence-quality gate ({}). Review the clip before acting.".format(
                        event_type.replace("_", " "), ", ".join(r.lower().replace("_", " ") for r in reasons) or "quality check failed"),
                    "evidence_reasons": reasons, "measurements": measurements or {}}
        return {"event_type": event_type, "timestamp_seconds": round(now, 2), "zone": zone,
                "track_id": track.id if track else None, "evidence_quality": quality,
                "reason_code": reason_code, "reason": reason, "evidence_reasons": reasons, "measurements": measurements or {}}

    def _assess(self, track, object_class=False, people=()):
        """Return (quality, reasons) for one track using observable image/track cues."""
        s = self.settings
        reasons = []
        width, height = track.box[2] - track.box[0], track.box[3] - track.box[1]
        if track.confidence < s.low_quality_min_confidence:
            reasons.append("LOW_DETECTOR_CONFIDENCE")
        min_height = s.low_quality_min_object_height if object_class else s.low_quality_min_person_height
        if height < min_height or width < 8:
            reasons.append("TARGET_TOO_SMALL")
        if not object_class and s.occlusion_iou > 0:
            overlap = max((box_iou(track.box, p.box) for p in people if p.id != track.id), default=0.0)
            if overlap >= s.occlusion_iou:
                reasons.append("HEAVY_OVERLAP_WITH_OTHER_PERSON")
        fq = self.frame_quality
        if fq:
            if s.low_light_brightness > 0 and fq.get("brightness") is not None and fq["brightness"] < s.low_light_brightness:
                reasons.append("LOW_LIGHT")
            if s.min_sharpness > 0 and fq.get("sharpness") is not None and fq["sharpness"] < s.min_sharpness:
                reasons.append("IMAGE_BLURRED")
        return ("SUFFICIENT" if not reasons else "INSUFFICIENT_EVIDENCE"), reasons

    def _quality(self, track, object_class=False):
        """Backward-compatible single-value quality check."""
        return self._assess(track, object_class)[0]

    # ------------------------------------------------------------------ main loop
    def update(self, tracks, frame_index, frame_shape, current_time=None, wall_clock_time=None, frame_quality=None, pose=None):
        now = current_time if current_time is not None else frame_index / self.fps
        self.update_count += 1
        self.frame_quality = frame_quality
        alerts = []
        people = [t for t in tracks if t.label == "person"]
        bags = [t for t in tracks if t.label in self.settings.bag_classes]
        for t in tracks:
            self.last_seen[t.id] = now
        for person in people:
            history = self.history.setdefault(person.id, deque(maxlen=300))
            history.append((now, center(person.box)))

        # Review alert for supported detector classes. This is object-label detection,
        # not a determination of intent, threat, or how the object is being used.
        s = self.settings
        for obj in tracks:
            if obj.label not in s.alert_object_classes:
                continue
            hits = self.object_hits[obj.id] = self.object_hits.get(obj.id, 0) + 1
            if obj.id in self.alerted_objects or hits < max(1, s.object_confirm_samples):
                continue
            self.alerted_objects.add(obj.id)
            if now - self.object_label_last_alert.get(obj.label, -1e9) < s.object_alert_cooldown_seconds:
                continue
            self.object_label_last_alert[obj.label] = now
            quality, reasons = self._assess(obj, object_class=True)
            alerts.append(self._alert("potentially_hazardous_object_detection", "MODEL_DETECTED_CONFIGURED_OBJECT_CLASS",
                "The detector labeled a visible object as '{}' in {} consecutive detections. Confirm the image evidence; the label does not establish how it is being used or whether anyone is at risk.".format(obj.label, hits),
                now, obj, quality=quality, reasons=reasons, measurements={"object_label": obj.label,
                "detector_confidence": round(obj.confidence, 3), "detections_before_alert": hits, "bounding_box_px": list(obj.box)}))

        alerts += self._zones(people, now, wall_clock_time)
        alerts += self._unattended(bags, people, now)
        if self.context and self.settings.enable_dwell_pacing:
            alerts += self._pacing(people, now)
        if self.context and self.settings.enable_counterflow:
            alerts += self._counterflow(people, now)
        if self.context and self.settings.enable_dispersion:
            alerts += self._dispersion(people, now)
        if self.settings.enable_tamper and frame_quality:
            alerts += self._tamper(frame_quality, now)
        if self.settings.enable_crowd:
            alerts += self._crowd(people, now)
        if self.settings.enable_running:
            alerts += self._running(people, now)
        if self.settings.enable_arms_raised and pose:
            alerts += self._arms_raised(people, pose, now)
        if self.update_count % 50 == 0:
            self._prune(now)
        return alerts

    # ------------------------------------------------------------------ core event 1
    def _zones(self, people, now, wall_clock_time):
        alerts = []
        for person in people:
            p = self.anchor(person.box)
            quality, reasons = self._assess(person, people=people)
            for zone in self.settings.restricted_zones:
                key = (person.id, zone.name)
                inside = point_in_zone(p, zone) if len(zone.polygon) >= 3 else False
                streak = self.inside_streak.get(key, 0) + 1 if inside else 0
                self.inside_streak[key] = streak
                confirmed = streak >= self.confirm_samples
                previous = self.previous_zone_state.get(key)
                if previous is None:
                    # First sighting: a person already inside is not reported as crossing.
                    if inside:
                        self.zone_entered_at[key] = now
                    else:
                        self.zone_seen_outside.add(key)
                    self.previous_zone_state[key] = inside
                    continue
                if not inside:
                    self.zone_seen_outside.add(key)
                if confirmed and not previous:
                    self.zone_entered_at[key] = now
                    last_alert = self.zone_last_alerted.get(key, -1e9)
                    if zone.alert_on_entry and key in self.zone_seen_outside and now-last_alert >= self.settings.zone_entry_cooldown_seconds:
                        active = self._schedule_active(zone, wall_clock_time) if self.context else True
                        measurements = {"anchor_point_px": list(p), "anchor": self.settings.zone_anchor,
                                        "detector_confidence": round(person.confidence, 3), "zone_type": zone.zone_type,
                                        "consecutive_inside_samples": streak}
                        if active is True:
                            alerts.append(self._alert("restricted_zone_entry", "PERSON_TRACK_CROSSED_CONFIGURED_BOUNDARY",
                                "A person track crossed into the configured restricted region '{}' and stayed inside for {} consecutive sample(s).".format(zone.name, streak),
                                now, person, zone.name, quality, measurements, reasons))
                        elif active is None:
                            alerts.append(self._alert("restricted_zone_entry", "ZONE_SCHEDULE_TIME_UNAVAILABLE",
                                "A zone entry was observed, but the configured schedule could not be checked for this video timestamp.", now,
                                person, zone.name, "INSUFFICIENT_EVIDENCE", measurements, reasons + ["SCHEDULE_TIME_UNAVAILABLE"]))
                        self.zone_last_alerted[key] = now
                elif confirmed and zone.enable_dwell and self.settings.enable_dwell_pacing and self.context:
                    started = self.zone_entered_at.get(key, now)
                    dwell = now - started
                    if dwell >= self.settings.dwell_seconds and key not in self.alerted_dwell:
                        alerts.append(self._alert("prolonged_zone_dwell", "CONFIGURED_ZONE_DWELL_THRESHOLD_REACHED",
                            "A person track remained in zone '{}' for {:.0f}s, beyond its {:.0f}s dwell threshold. Waiting is not inherently suspicious.".format(zone.name, dwell, self.settings.dwell_seconds),
                            now, person, zone.name, quality, {"dwell_seconds": round(dwell, 2)}, reasons))
                        self.alerted_dwell.add(key)
                if not inside:
                    self.zone_entered_at.pop(key, None)
                    self.previous_zone_state[key] = False
                elif confirmed:
                    self.previous_zone_state[key] = True
        return alerts

    # ------------------------------------------------------------------ core event 2
    def _unattended(self, bags, people, now):
        s = self.settings
        alerts = []
        for bag in bags:
            bcenter = center(bag.box)
            distances = [(hypot(bcenter[0]-center(p.box)[0], bcenter[1]-center(p.box)[1]), p) for p in people]
            distances.sort(key=lambda x: x[0])
            nearest = distances[0] if distances else None
            near = bool(nearest and nearest[0] < s.association_distance_px)
            state = self.objects.get(bag.id)
            if state is None:
                state = {"center": bcenter, "last_moved": now, "near_person": near, "separated_since": None,
                         "alerted": False, "first_update": self.update_count, "seen_updates": 0,
                         "stationary_first_update": self.update_count, "stationary_seen": 0, "associated_ids": set()}
                self.objects[bag.id] = state
            state["seen_updates"] += 1
            moved = hypot(bcenter[0]-state["center"][0], bcenter[1]-state["center"][1])
            if moved > max(3.0, s.stationary_pixels_per_second / self.fps):
                state["center"], state["last_moved"] = bcenter, now
                state["stationary_first_update"], state["stationary_seen"] = self.update_count, 0
            state["stationary_seen"] += 1
            if near:
                state["near_person"], state["separated_since"] = True, None
                state["associated_ids"].update(p.id for d, p in distances if d < s.association_distance_px)
            elif state["near_person"] and state["separated_since"] is None:
                state["separated_since"] = now
            dwell = now - state["last_moved"]
            if state["alerted"] or dwell < s.unattended_seconds:
                continue
            if self.context:
                separated = state["separated_since"] is not None and now - state["separated_since"] >= s.separation_seconds
                if not separated:
                    continue
                quality, reasons = self._assess(bag, object_class=True)
                expected = self.update_count - state["stationary_first_update"] + 1
                visible = state["stationary_seen"] / expected if expected else 1.0
                if visible < s.object_min_visible_fraction:
                    reasons.append("OBJECT_TRACK_FRAGMENTED")
                borderline = [d for d, _ in distances if s.association_distance_px <= d < s.association_distance_px * s.association_ambiguity_factor]
                if borderline:
                    reasons.append("PERSON_ASSOCIATION_AMBIGUOUS")
                quality = "SUFFICIENT" if not reasons else "INSUFFICIENT_EVIDENCE"
                state["alerted"] = True
                alerts.append(self._alert("potentially_unattended_object", "STATIONARY_OBJECT_AFTER_PERSON_SEPARATION",
                    "A {} stayed stationary for {:.0f}s and the nearby person track(s) moved away {:.0f}s ago. Ownership is a proximity guess and is uncertain.".format(
                        bag.label, dwell, now - state["separated_since"]),
                    now, bag, quality=quality, reasons=reasons, measurements={"stationary_seconds": round(dwell, 2),
                    "seconds_since_separation": round(now - state["separated_since"], 2),
                    "nearest_person_distance_px": round(nearest[0], 1) if nearest else None,
                    "previously_nearby_tracks": len(state["associated_ids"]),
                    "object_visible_fraction": round(visible, 3), "detector_confidence": round(bag.confidence, 3)}))
            else:
                state["alerted"] = True
                quality, reasons = self._assess(bag, object_class=True)
                alerts.append(self._alert("potentially_unattended_object", "STATIONARY_OBJECT_DURATION",
                    "A {} stayed stationary for {:.0f}s (single fixed rule; no person context).".format(bag.label, dwell),
                    now, bag, quality=quality, reasons=reasons, measurements={"stationary_seconds": round(dwell, 2),
                    "nearest_person_distance_px": round(nearest[0], 1) if nearest else None,
                    "detector_confidence": round(bag.confidence, 3)}))
        return alerts

    # ------------------------------------------------------------------ extensions
    def _pacing(self, people, now):
        alerts = []
        window = self.settings.pacing_window_seconds
        for person in people:
            if person.id in self.alerted_pacing:
                continue
            h = [x for x in self.history.get(person.id, []) if now - x[0] <= window]
            if len(h) < 5:
                continue
            # Displacement-based: a direction is only established after moving pacing_min_step_px,
            # so detector jitter on a standing person does not count as pacing.
            reversals, prior, anchor = 0, None, h[0][1]
            for _, point in h[1:]:
                step = vector(anchor, point)
                if magnitude(step) < self.settings.pacing_min_step_px:
                    continue
                if prior is not None and cosine(prior, step) < -0.55:
                    reversals += 1
                prior, anchor = step, point
            if reversals >= self.settings.pacing_reversals:
                quality, reasons = self._assess(person, people=people)
                alerts.append(self._alert("repeated_path_reversals", "REPEATED_TRACK_DIRECTION_REVERSALS",
                    "A track reversed direction {} times within {:.0f}s. Repeated walking back and forth is not inherently suspicious.".format(reversals, now - h[0][0]),
                    now, person, quality=quality, reasons=reasons, measurements={"reversals": reversals,
                    "window_seconds": round(now - h[0][0], 2)}))
                self.alerted_pacing.add(person.id)
        return alerts

    def _velocity(self, person, lookback=1.0):
        """Smoothed velocity over ~lookback seconds of history (px/s)."""
        h = self.history.get(person.id, [])
        if len(h) < 2:
            return None
        end = h[-1]
        start = next((x for x in h if end[0] - x[0] <= lookback), h[-2])
        if start is end:
            start = h[-2]
        dt = end[0] - start[0]
        if dt <= 0:
            return None
        v = vector(start[1], end[1])
        return (v[0]/dt, v[1]/dt)

    def _counterflow(self, people, now):
        alerts = []
        s = self.settings
        if len(people) < s.counterflow_min_crowd_tracks:
            return alerts
        velocities = {}
        for person in people:
            v = self._velocity(person)
            if v is not None and magnitude(v) >= s.counterflow_min_speed_px_s:
                velocities[person.id] = v
        for person in people:
            v = velocities.get(person.id)
            peers = [other for ident, other in velocities.items() if ident != person.id]
            if v is None or len(peers) < max(1, s.counterflow_min_crowd_tracks - 1):
                self.counterflow_since.pop(person.id, None)
                continue
            # Unit-vector mean so one fast walker does not define the crowd direction.
            units = [(x[0]/magnitude(x), x[1]/magnitude(x)) for x in peers]
            mean = (sum(u[0] for u in units), sum(u[1] for u in units))
            coherence = magnitude(mean) / len(units)
            if coherence >= s.counterflow_min_coherence and cosine(v, mean) <= s.counterflow_cosine_threshold:
                self.counterflow_since.setdefault(person.id, now)
                sustained = now - self.counterflow_since[person.id]
                if sustained >= s.counterflow_sustain_seconds and person.id not in self.alerted_counterflow:
                    quality, reasons = self._assess(person, people=people)
                    alerts.append(self._alert("sustained_counter_flow", "TRACK_MOVES_AGAINST_LOCAL_FLOW",
                        "A track moved against the dominant direction of {} nearby moving tracks for {:.1f}s.".format(len(peers), sustained), now,
                        person, quality=quality, reasons=reasons, measurements={"sustained_seconds": round(sustained, 2),
                        "local_flow_tracks": len(peers), "flow_coherence": round(coherence, 3)}))
                    self.alerted_counterflow.add(person.id)
            else:
                self.counterflow_since.pop(person.id, None)
        return alerts

    def _dispersion(self, people, now):
        s = self.settings
        alerts = []
        candidates = []
        if len(people) >= s.dispersion_min_people:
            centers = [center(p.box) for p in people]
            centroid = (sum(p[0] for p in centers)/len(centers), sum(p[1] for p in centers)/len(centers))
            for person, c in zip(people, centers):
                v = self._velocity(person, lookback=0.5)
                if v is None:
                    continue
                radial = (c[0]-centroid[0], c[1]-centroid[1])
                if magnitude(v) >= s.dispersion_speed_px_s and cosine(v, radial) > 0.15:
                    candidates.append(person)
        if len(candidates) >= max(2, int(s.dispersion_min_people * 0.6)):
            self.dispersion_since = self.dispersion_since if self.dispersion_since is not None else now
            if now-self.dispersion_since >= s.dispersion_sustain_seconds and not self.dispersion_alerted:
                alerts.append(self._alert("sudden_group_dispersion", "MULTIPLE_TRACKS_MOVE_OUTWARD",
                    "{} of {} tracks moved outward from the group centre at speed. This describes motion only and does not imply panic or danger.".format(len(candidates), len(people)), now,
                    quality="SUFFICIENT", measurements={"outward_tracks": len(candidates), "group_tracks": len(people),
                    "sustained_seconds": round(now-self.dispersion_since, 2)}))
                self.dispersion_alerted = True
        else:
            self.dispersion_since, self.dispersion_alerted = None, False
        return alerts

    # ------------------------------------------------------------------ demo-oriented cues
    def _tamper(self, fq, now):
        s = self.settings
        cues = []
        if fq.get("brightness") is not None and fq["brightness"] < s.tamper_dark_brightness:
            cues.append("VIEW_DARK")
        if fq.get("sharpness") is not None and fq["sharpness"] < s.tamper_min_sharpness:
            cues.append("VIEW_FEATURELESS")
        moved = fq.get("changed_fraction") is not None and fq["changed_fraction"] >= s.tamper_changed_fraction
        # A person stepping close to the lens also changes most of the view, so "changed" alone
        # only counts when tamper_detect_moved is on; otherwise it is a supporting cue.
        if moved and (cues or s.tamper_detect_moved):
            cues.append("MOST_OF_VIEW_CHANGED")
        if not cues:
            if self.tamper_since is not None:
                self.tamper_clear_since = self.tamper_clear_since if self.tamper_clear_since is not None else now
                if now - self.tamper_clear_since >= 1.0:
                    self.tamper_since, self.tamper_alerted, self.tamper_clear_since = None, False, None
            return []
        self.tamper_clear_since = None
        self.tamper_since = self.tamper_since if self.tamper_since is not None else now
        held = now - self.tamper_since
        if held < s.tamper_sustain_seconds or self.tamper_alerted:
            return []
        self.tamper_alerted = True
        return [self._alert("camera_tampering_suspected", "CAMERA_VIEW_OBSTRUCTED_OR_MOVED",
            "The camera view stayed {} for {:.1f}s (lens covered, blocked, defocused or moved). Check the camera.".format(
                " and ".join(c.lower().replace("_", " ") for c in cues), held), now,
            measurements={"cues": cues, "sustained_seconds": round(held, 2),
                          "brightness": round(fq.get("brightness") or 0, 1), "sharpness": round(fq.get("sharpness") or 0, 1),
                          "changed_fraction": round(fq.get("changed_fraction") or 0, 3)})]

    def _crowd(self, people, now):
        s = self.settings
        if len(people) < s.crowd_alert_count:
            self.crowd_since, self.crowd_alerted = None, False
            return []
        self.crowd_since = self.crowd_since if self.crowd_since is not None else now
        if now - self.crowd_since < s.crowd_sustain_seconds or self.crowd_alerted:
            return []
        self.crowd_alerted = True
        return [self._alert("crowding_threshold_exceeded", "PERSON_COUNT_ABOVE_THRESHOLD",
            "{} people were visible for {:.0f}s, at or above the configured limit of {}.".format(len(people), now - self.crowd_since, s.crowd_alert_count),
            now, measurements={"people": len(people), "threshold": s.crowd_alert_count})]

    def _running(self, people, now):
        s = self.settings
        alerts = []
        for person in people:
            v = self._velocity(person, lookback=0.5)
            height = max(1, person.box[3] - person.box[1])
            rate = magnitude(v) / height if v else 0.0
            if rate < s.running_body_heights_per_s:
                self.running_since.pop(person.id, None)
                continue
            start = self.running_since.setdefault(person.id, now)
            if now - start >= s.running_sustain_seconds and person.id not in self.alerted_running:
                self.alerted_running.add(person.id)
                quality, reasons = self._assess(person, people=people)
                alerts.append(self._alert("rapid_movement", "TRACK_SPEED_ABOVE_THRESHOLD",
                    "A person track moved at {:.1f} body-heights per second for {:.1f}s. Fast movement is not inherently suspicious.".format(rate, now - start),
                    now, person, quality=quality, reasons=reasons,
                    measurements={"body_heights_per_second": round(rate, 2), "sustained_seconds": round(now - start, 2)}))
        return alerts

    def _arms_raised(self, people, pose, now):
        s = self.settings
        alerts = []
        for person in people:
            raised = (pose.get(person.id) or {}).get("arms_raised", 0)
            if raised < 2:
                self.arms_since.pop(person.id, None)
                continue
            start = self.arms_since.setdefault(person.id, now)
            if now - start >= s.arms_raised_sustain_seconds and person.id not in self.alerted_arms:
                self.alerted_arms.add(person.id)
                quality, reasons = self._assess(person, people=people)
                alerts.append(self._alert("both_arms_raised", "POSE_WRISTS_ABOVE_SHOULDERS",
                    "Both wrists stayed above shoulder height for {:.1f}s. This is a posture cue only; check the context.".format(now - start),
                    now, person, quality=quality, reasons=reasons, measurements={"sustained_seconds": round(now - start, 2),
                    "visible_keypoints": (pose.get(person.id) or {}).get("visible_keypoints")}))
        return alerts

    # ------------------------------------------------------------------ housekeeping
    def _prune(self, now):
        """Drop per-track state for tracks not seen recently (keeps long live runs bounded)."""
        ttl = self.settings.state_ttl_seconds
        stale = {i for i, t in self.last_seen.items() if now - t > ttl}
        if not stale:
            return
        for i in stale:
            self.last_seen.pop(i, None); self.history.pop(i, None); self.counterflow_since.pop(i, None)
            self.objects.pop(i, None); self.object_hits.pop(i, None)
            self.running_since.pop(i, None); self.arms_since.pop(i, None)
        for table in (self.previous_zone_state, self.inside_streak, self.zone_entered_at, self.zone_last_alerted):
            for key in [k for k in table if k[0] in stale]:
                table.pop(key, None)
        self.zone_seen_outside = {k for k in self.zone_seen_outside if k[0] not in stale}

    def _schedule_active(self, zone, wall_clock_time):
        if not zone.active_hours:
            return True
        if wall_clock_time is None:
            return None
        current = wall_clock_time.astimezone(ZoneInfo(self.settings.schedule_timezone)).time()
        for window in zone.active_hours:
            try:
                start_text, end_text = window.split("-", 1)
                start = clock_time.fromisoformat(start_text); end = clock_time.fromisoformat(end_text)
            except ValueError:
                continue
            if (start <= current <= end) if start <= end else (current >= start or current <= end):
                return True
        return False
