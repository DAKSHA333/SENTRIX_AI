from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Point = Tuple[int, int]


@dataclass
class Zone:
    name: str
    polygon: List[Point]
    alert_on_entry: bool = True
    zone_type: str = "restricted"
    enable_dwell: bool = False
    active_hours: List[str] = field(default_factory=list)


@dataclass
class Settings:
    model: str = "yolov8n.pt"
    confidence: float = 0.35
    sample_every: int = 3
    min_box_area: int = 500
    max_track_gap: int = 12
    match_iou: float = 0.15
    motion_area_fraction: float = 0.003
    motion_pixel_threshold: int = 25
    motion_cooldown_seconds: float = 2.0
    tracker_backend: str = "bytetrack"
    tracker_config: str = "bytetrack.yaml"
    restricted_zones: List[Zone] = field(default_factory=list)
    bag_classes: Tuple[str, ...] = ("backpack", "handbag", "suitcase")
    alert_object_classes: Tuple[str, ...] = ("knife", "scissors")
    stationary_pixels_per_second: float = 3.0
    unattended_seconds: float = 20.0
    association_distance_px: float = 140.0
    zone_entry_cooldown_seconds: float = 5.0
    # "bottom" tests the box's foot point (ground plane); "center" keeps the legacy torso centre.
    zone_anchor: str = "bottom"
    # Context mode only: consecutive inside samples before an entry is confirmed (hysteresis).
    zone_entry_confirm_samples: int = 2
    low_quality_min_person_height: int = 35
    low_quality_min_object_height: int = 12
    low_quality_min_confidence: float = 0.25
    occlusion_iou: float = 0.5
    low_light_brightness: float = 35.0
    min_sharpness: float = 0.0
    separation_seconds: float = 2.0
    object_min_visible_fraction: float = 0.6
    association_ambiguity_factor: float = 1.5
    state_ttl_seconds: float = 60.0
    evidence_max_width: int = 960
    blur_faces_in_evidence: bool = True
    evidence_dir: str = "data/evidence"
    evidence_pre_seconds: float = 3.0
    evidence_post_seconds: float = 3.0
    retention_days: int = 7
    trajectory_csv: str = "data/trajectories.csv"
    pose_enabled: bool = False
    pose_model: str = "yolov8n-pose.pt"
    enable_dwell_pacing: bool = False
    dwell_seconds: float = 300.0
    pacing_reversals: int = 3
    pacing_window_seconds: float = 30.0
    pacing_min_step_px: float = 25.0
    enable_counterflow: bool = False
    counterflow_cosine_threshold: float = -0.65
    counterflow_min_speed_px_s: float = 20.0
    counterflow_sustain_seconds: float = 3.0
    counterflow_min_crowd_tracks: int = 3
    counterflow_min_coherence: float = 0.6
    enable_dispersion: bool = False
    dispersion_min_people: int = 4
    dispersion_speed_px_s: float = 70.0
    dispersion_sustain_seconds: float = 1.0
    # Object-class alerts: detections needed per track before alerting, and per-label cooldown.
    object_confirm_samples: int = 1
    object_alert_cooldown_seconds: float = 0.0
    # Optional second detector for classes COCO lacks (e.g. a custom-trained firearm model).
    weapon_model: str = ""
    weapon_confidence: float = 0.4
    device: str = ""
    # Camera tampering: lens covered/dark, featureless (blurred/blocked), or most of the view changed.
    enable_tamper: bool = False
    tamper_dark_brightness: float = 12.0
    tamper_min_sharpness: float = 4.0
    tamper_changed_fraction: float = 0.75
    tamper_sustain_seconds: float = 2.0
    tamper_detect_moved: bool = False
    # Crowding: at least N people for a sustained period.
    enable_crowd: bool = False
    crowd_alert_count: int = 5
    crowd_sustain_seconds: float = 3.0
    # Running: speed in body-heights per second (scale-independent), sustained.
    enable_running: bool = False
    running_body_heights_per_s: float = 1.6
    running_sustain_seconds: float = 1.0
    # Both arms raised (pose model required), sustained. A posture cue only.
    enable_arms_raised: bool = False
    arms_raised_sustain_seconds: float = 2.0
    database: str = "data/sentrix.db"
    schedule_timezone: str = "Asia/Kolkata"

    @classmethod
    def from_json(cls, path: str) -> "Settings":
        import json
        from datetime import time
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        raw["restricted_zones"] = [Zone(**z) for z in raw.get("restricted_zones", [])]
        for key in ("bag_classes", "alert_object_classes"):
            if key in raw:
                raw[key] = tuple(raw[key])
        known = set(cls.__dataclass_fields__)
        unknown = sorted(k for k in raw if k not in known and not k.startswith("_"))
        if unknown:
            raise ValueError("Unknown config keys in {}: {}".format(path, unknown))
        raw = {k: v for k, v in raw.items() if not k.startswith("_")}
        settings=cls(**raw)
        if not 0.0<settings.confidence<=1.0:raise ValueError("confidence must be between 0 and 1")
        if settings.sample_every<1 or settings.unattended_seconds<=0:raise ValueError("sample_every and unattended_seconds must be positive")
        if settings.tracker_backend not in ("bytetrack","iou"):raise ValueError("tracker_backend must be bytetrack or iou")
        if settings.zone_anchor not in ("bottom","center"):raise ValueError("zone_anchor must be bottom or center")
        for zone in settings.restricted_zones:
            if zone.polygon and len(zone.polygon)<3:raise ValueError("zone {} needs at least 3 polygon points".format(zone.name))
            for window in zone.active_hours:
                try:
                    start,end=window.split("-",1);time.fromisoformat(start);time.fromisoformat(end)
                except (ValueError,TypeError):raise ValueError("Invalid active_hours window {!r} for zone {}".format(window,zone.name))
        return settings
