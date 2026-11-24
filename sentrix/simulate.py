"""Synthetic scenario generator for pipeline verification -- NOT a research dataset.

Renders simple top-down-ish scenes (rectangle "people" and "bags"), writes:
  * <clip>.mp4                 rendered video (so the motion baseline and evidence clips run)
  * <clip>.tracks.json         noisy simulated detector+tracker output (replayed with --detections)
  * manifest.csv / annotations.csv with ground truth derived from the scripted motion
  * config.synthetic.json      the rule configuration used for the scenes

The point is to exercise every stage end to end (events, evidence gate, storage,
evidence clips, export, benchmark, comparison, trajectory features) before consented
footage exists, and to act as a regression test. Numbers from this data say nothing
about real-world performance and must never be reported as results.
"""
import argparse
import csv
import json
import os
import random
import zlib
from math import hypot
import cv2
import numpy as np

W, H, FPS = 640, 360, 10
ZONE = [[420, 170], [625, 170], [625, 352], [420, 352]]
SCENARIOS = ("entry", "boundary_linger", "bag_abandon", "bag_benign_stay", "bag_brief", "entry_lowvis",
             "crowd_walk", "flicker_empty", "pacing", "counterflow", "dispersion")

SIM_CONFIG = {
    "_comment": "Synthetic-scene settings. Pixel thresholds are scaled to the 640x360 synthetic canvas, not a real camera.",
    "confidence": 0.25, "sample_every": 1, "min_box_area": 200,
    "restricted_zones": [{"name": "restricted_sim", "polygon": ZONE, "alert_on_entry": True, "zone_type": "restricted"}],
    "unattended_seconds": 8.0, "association_distance_px": 90.0, "separation_seconds": 2.0,
    "low_quality_min_person_height": 35, "low_quality_min_object_height": 12, "low_quality_min_confidence": 0.3,
    "zone_entry_confirm_samples": 2, "zone_entry_cooldown_seconds": 5.0,
    "enable_dwell_pacing": True, "pacing_reversals": 3, "pacing_window_seconds": 30.0, "pacing_min_step_px": 25.0,
    "enable_counterflow": True, "counterflow_min_speed_px_s": 20.0, "counterflow_sustain_seconds": 3.0, "counterflow_min_crowd_tracks": 3,
    "enable_dispersion": True, "dispersion_min_people": 4, "dispersion_speed_px_s": 90.0, "dispersion_sustain_seconds": 1.0,
    "motion_area_fraction": 0.004, "evidence_pre_seconds": 2.0, "evidence_post_seconds": 2.0,
    "alert_object_classes": [], "retention_days": 7,
}


# ---------------------------------------------------------------------------- actors
class Actor:
    def __init__(self, ident, label, size, waypoints=None, t_on=0.0, t_off=1e9, conf=0.85, color=(90, 90, 160)):
        self.id, self.label, self.size = ident, label, size
        self.waypoints = waypoints or []
        self.t_on, self.t_off, self.conf, self.color = t_on, t_off, conf, color
        self.carrier, self.attached, self.offset = None, [], (0, 0)

    def pos(self, t):
        if self.carrier is not None:
            return self._carried_pos(t)
        wp = self.waypoints
        if t <= wp[0][0]:
            return wp[0][1], wp[0][2]
        for (t0, x0, y0), (t1, x1, y1) in zip(wp, wp[1:]):
            if t0 <= t <= t1:
                a = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
                return x0 + a*(x1-x0), y0 + a*(y1-y0)
        return wp[-1][1], wp[-1][2]

    def _carried_pos(self, t):
        ref = None
        for a, b in self.attached:
            if a <= t <= b:
                ref = t; break
            if b < t:
                ref = b
        if ref is None:
            ref = self.attached[0][0]
        x, y = self.carrier.pos(ref)
        return x + self.offset[0], y + self.offset[1]

    def box(self, t):
        x, y = self.pos(t)
        w, h = self.size
        return (x - w/2, y - h, x + w/2, y)

    def visible(self, t):
        if not (self.t_on <= t < self.t_off):
            return False
        x1, y1, x2, y2 = self.box(t)
        return x2 > 0 and x1 < W and y2 > 0 and y1 < H


def bag_for(ident, carrier, attached, offset=(20, 0), size=(24, 26)):
    bag = Actor(ident, "backpack", size, conf=0.7, color=(30, 50, 80))
    bag.carrier, bag.attached, bag.offset = carrier, attached, offset
    bag.t_on, bag.t_off = carrier.t_on, 1e9
    return bag


def person(ident, waypoints, rng, size=None, **kw):
    size = size or (int(rng.uniform(36, 44)), int(rng.uniform(100, 118)))
    color = tuple(int(c) for c in rng.choice([(150, 60, 60), (60, 120, 60), (70, 70, 150), (120, 110, 40), (130, 70, 130), (60, 120, 140)]))
    return Actor(ident, "person", size, waypoints, color=color, **kw)


def inside(point):
    return cv2.pointPolygonTest(np.array(ZONE, dtype="int32"), (float(point[0]), float(point[1])), False) >= 0


def first_entry(actor, start, end):
    was_out = False
    t = start
    while t <= end:
        if actor.visible(t):
            if inside(actor.pos(t)):
                if was_out:
                    return round(t, 2)
            else:
                was_out = True
        t += 1.0/FPS
    return None


def centre(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


# ---------------------------------------------------------------------------- scenarios
def build(kind, seed):
    rng = random.Random(zlib.crc32("{}:{}".format(kind, seed).encode()))
    S = SIM_CONFIG
    actors, events, opts = [], [], {"illumination": "normal", "crowd_density": "low", "dim": 1.0, "flicker": False, "sway": rng.random() < 0.5}
    dur = 22.0
    if kind == "entry":
        y = rng.uniform(250, 340); sp = rng.uniform(45, 70); t0 = rng.uniform(0.5, 3)
        p = person(1, [(t0, -30, y), (t0 + (520+30)/sp, 520, y), (dur, 520, y)], rng); actors.append(p)
        trig = first_entry(p, 0, dur)
        events.append(dict(event_type="restricted_zone_entry", start=trig, end=trig+2.0, trigger=trig, track_id=1))
    elif kind == "boundary_linger":
        # Hard negative: walks up to the zone edge and stands with feet just outside it.
        y = rng.uniform(240, 330); stop_x = ZONE[0][0] - rng.uniform(3, 6); sp = rng.uniform(45, 65)
        t_arrive = 1 + (stop_x+30)/sp
        p = person(1, [(1, -30, y), (t_arrive, stop_x, y), (t_arrive+9, stop_x, y), (t_arrive+9+(stop_x+40)/sp, -40, y)], rng); actors.append(p)
    elif kind in ("bag_abandon", "bag_benign_stay", "bag_brief"):
        y = rng.uniform(200, 300); sp = rng.uniform(45, 65); drop_x = rng.uniform(180, 320)
        t_drop = 1 + (drop_x+30)/sp
        if kind == "bag_abandon":
            exit_t = t_drop + 0.8 + (drop_x+40)/sp
            p = person(1, [(1, -30, y), (t_drop, drop_x, y), (t_drop+0.8, drop_x, y), (exit_t, -40, y)], rng)
            bag = bag_for(101, p, [(0, t_drop)]); actors += [p, bag]
            t = t_drop
            while hypot(*np.subtract(centre(p.box(t)), centre(bag.box(t)))) < S["association_distance_px"] and t < dur:
                t += 1.0/FPS
            trig = round(max(t_drop + S["unattended_seconds"], t + S["separation_seconds"]), 2)
            dur = max(dur, trig + 6)
            events.append(dict(event_type="potentially_unattended_object", start=trig, end=dur, trigger=trig, track_id=101))
        elif kind == "bag_benign_stay":
            stay = S["unattended_seconds"] + rng.uniform(4, 7)
            p = person(1, [(1, -30, y), (t_drop, drop_x, y), (t_drop+0.6, drop_x+28, y), (t_drop+stay, drop_x+28, y),
                           (t_drop+stay+0.6, drop_x, y), (t_drop+stay+0.6+(drop_x+40)/sp, -40, y)], rng)
            bag = bag_for(101, p, [(0, t_drop), (t_drop+stay+0.6, 1e9)]); actors += [p, bag]
            dur = max(dur, t_drop + stay + 6)
        else:
            # Benign look-alike: bag is on the floor for less than unattended_seconds in total.
            away = rng.uniform(0.8, 1.6); step = 120
            far = drop_x + step
            go = step / sp
            p = person(1, [(1, -30, y), (t_drop, drop_x, y), (t_drop+go, far, y), (t_drop+go+away, far, y),
                           (t_drop+2*go+away, drop_x, y), (t_drop+2*go+away+(drop_x+40)/sp, -40, y)], rng)
            back = t_drop+2*go+away
            bag = bag_for(101, p, [(0, t_drop), (back, 1e9)]); actors += [p, bag]
            dur = max(dur, back + 6)
    elif kind == "entry_lowvis":
        opts.update(illumination="dim", dim=0.22)
        y = rng.uniform(200, 240); sp = rng.uniform(30, 45)
        p = person(1, [(1, -20, y), (1+(500+20)/sp, 500, y), (dur, 500, y)], rng, size=(12, 30), conf=0.33); actors.append(p)
        trig = first_entry(p, 0, dur)
        events.append(dict(event_type="restricted_zone_entry", start=trig, end=trig+2.0, trigger=trig, track_id=1,
                           low_visibility=True, visibility_reason="small distant person in poor light"))
    elif kind == "crowd_walk":
        opts["crowd_density"] = "medium"
        for i in range(5):
            y = rng.uniform(110, 160); sp = rng.uniform(40, 65); t0 = rng.uniform(0, 8)
            if rng.random() < 0.5:
                wp = [(t0, -30, y), (t0+(W+60)/sp, W+30, y)]
            else:
                wp = [(t0, W+30, y), (t0+(W+60)/sp, -30, y)]
            actors.append(person(i+1, wp, rng))
    elif kind == "flicker_empty":
        opts.update(illumination="flicker", flicker=True, sway=True)
    elif kind == "pacing":
        y = rng.uniform(150, 200); a, b = rng.uniform(60, 120), rng.uniform(300, 360); sp = rng.uniform(50, 70)
        leg = (b - a) / sp
        wp = [(0.5, a, y)]
        turns = []
        for k in range(5):
            t_prev = wp[-1][0]
            x = b if k % 2 == 0 else a
            wp.append((t_prev + leg, x, y))
            turns.append(t_prev + leg)
        p = person(1, wp, rng); actors.append(p)
        trig = round(turns[S["pacing_reversals"]-1] + S["pacing_min_step_px"]/sp, 2)
        dur = max(dur, wp[-1][0] + 2)
        events.append(dict(event_type="repeated_path_reversals", start=round(turns[0], 2), end=trig+3.0, trigger=trig, track_id=1))
    elif kind == "counterflow":
        opts["crowd_density"] = "medium"
        for i in range(4):
            y = 95 + i*20 + rng.uniform(-4, 4); sp = rng.uniform(45, 60); t0 = rng.uniform(0, 2)
            actors.append(person(i+1, [(t0, -40 - i*25, y), (t0 + (W+120)/sp, W+80, y)], rng))
        y = rng.uniform(115, 145); sp = rng.uniform(45, 60); t0 = rng.uniform(2.5, 4)
        c = person(9, [(t0, W+30, y), (t0+(W+70)/sp, -40, y)], rng); actors.append(c)
        trig = round(t0 + S["counterflow_sustain_seconds"] + 1.0, 2)
        events.append(dict(event_type="sustained_counter_flow", start=round(t0, 2), end=round(t0+(W)/sp, 2), trigger=trig, track_id=9))
    elif kind == "dispersion":
        opts["crowd_density"] = "high"
        cx, cy = rng.uniform(200, 300), rng.uniform(200, 240); t_d = rng.uniform(7, 10)
        n = 6
        for i in range(n):
            ang = 2*np.pi*i/n + rng.uniform(-0.3, 0.3)
            x0, y0 = cx + 55*np.cos(ang), cy + 30*np.sin(ang)
            sp = rng.uniform(130, 170)
            x1, y1 = x0 + sp*4*np.cos(ang), y0 + sp*4*0.6*np.sin(ang)
            actors.append(person(i+1, [(0, x0, y0), (t_d, x0, y0), (t_d+4, x1, y1)], rng))
        trig = round(t_d + S["dispersion_sustain_seconds"] + 0.5, 2)
        events.append(dict(event_type="sudden_group_dispersion", start=round(t_d, 2), end=round(t_d+4, 2), trigger=trig, track_id=""))
    else:
        raise ValueError(kind)
    labelled = {(e["event_type"], e.get("track_id")) for e in events}
    for a in actors:
        if a.label == "person" and ("restricted_zone_entry", a.id) not in labelled:
            trig = first_entry(a, 0, dur)
            if trig is not None:
                events.append(dict(event_type="restricted_zone_entry", start=trig, end=trig+2.0, trigger=trig, track_id=a.id))
    return actors, events, opts, round(dur, 1)


# ---------------------------------------------------------------------------- rendering
def background(rng):
    bg = np.full((H, W, 3), 128, np.uint8)
    for y in range(0, H, 40):
        for x in range(0, W, 40):
            v = 118 + ((x//40 + y//40) % 2) * 14 + rng.randint(-3, 3)
            bg[y:y+40, x:x+40] = (v, v, v+4)
    cv2.rectangle(bg, (0, 0), (W, 70), (95, 100, 105), -1)          # far wall
    cv2.rectangle(bg, (30, 20), (150, 62), (70, 80, 90), -1)        # notice board
    cv2.rectangle(bg, (470, 15), (560, 70), (60, 60, 70), -1)       # door
    pts = np.array(ZONE, np.int32)
    for i in range(4):                                              # yellow floor tape
        a, b = pts[i], pts[(i+1) % 4]
        n = int(hypot(*(b-a)) // 14)
        for k in range(0, n, 2):
            p0 = a + (b-a)*k/n; p1 = a + (b-a)*(k+1)/n
            cv2.line(bg, tuple(p0.astype(int)), tuple(p1.astype(int)), (40, 200, 230), 3)
    return bg


def draw_actor(frame, actor, t):
    x1, y1, x2, y2 = [int(v) for v in actor.box(t)]
    w, h = x2-x1, y2-y1
    if actor.label == "person":
        cx = (x1+x2)//2
        cv2.rectangle(frame, (x1+int(.1*w), y1+int(.22*h)), (x2-int(.1*w), y1+int(.62*h)), actor.color, -1)
        phase = int(t*4) % 2
        lw = max(2, w//4)
        cv2.rectangle(frame, (cx-lw-1, y1+int(.62*h)), (cx-1, y2 - (3 if phase else 0)), (50, 50, 60), -1)
        cv2.rectangle(frame, (cx+1, y1+int(.62*h)), (cx+lw+1, y2 - (0 if phase else 3)), (50, 50, 60), -1)
        cv2.circle(frame, (cx, y1+int(.11*h)), max(3, int(.11*h)), (120, 150, 185), -1)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), actor.color, -1)
        cv2.rectangle(frame, (x1+w//4, y1-4), (x2-w//4, y1+2), (20, 30, 50), 2)


def render_and_track(kind, seed, out_dir, clip_id):
    actors, events, opts, dur = build(kind, seed)
    rng = random.Random(seed * 7919 + len(kind))
    nrng = np.random.default_rng(seed * 104729 + len(kind))
    bg = background(rng)
    noise_bank = [nrng.normal(0, 3, (H, W, 3)).astype(np.float32) for _ in range(6)]
    n_frames = int(dur * FPS)
    video = os.path.join(out_dir, clip_id + ".mp4")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    frames = {}
    gain, next_flip = 1.0, 2.0
    for f in range(n_frames):
        t = f / FPS
        frame = bg.copy()
        if opts["sway"]:
            sx = int(590 + 10*np.sin(t*2.2))
            cv2.ellipse(frame, (sx, 95), (16, 26), 0, 0, 360, (60, 110, 60), -1)
        order = sorted((a for a in actors if a.visible(t)), key=lambda a: a.box(t)[3])
        for a in order:
            draw_actor(frame, a, t)
        if opts["flicker"] and t >= next_flip:
            gain = 0.72 if gain == 1.0 else 1.0
            next_flip = t + rng.uniform(1.5, 4.0)
        g = opts["dim"] * gain
        img = frame.astype(np.float32) * g + noise_bank[(f * 5 + int(nrng.integers(6))) % 6]
        writer.write(np.clip(img, 0, 255).astype(np.uint8))
        dets = []
        for a in actors:
            if not a.visible(t) or rng.random() < 0.04:
                continue
            x1, y1, x2, y2 = a.box(t)
            j = 0.03 * (y2 - y1)
            box = [int(round(v + rng.gauss(0, j))) for v in (x1, y1, x2, y2)]
            box = [max(0, min(W-1, box[0])), max(0, min(H-1, box[1])), max(0, min(W-1, box[2])), max(0, min(H-1, box[3]))]
            if box[2]-box[0] < 4 or box[3]-box[1] < 4:
                continue
            conf = round(min(0.99, max(0.05, a.conf + rng.gauss(0, 0.04))), 3)
            dets.append([a.id, a.label, *box, conf])
        if dets:
            frames[str(f)] = dets
    writer.release()
    with open(os.path.join(out_dir, clip_id + ".tracks.json"), "w", encoding="utf-8") as handle:
        json.dump({"source": "sentrix.simulate (synthetic, noisy oracle tracker)", "fps": FPS, "frames": frames}, handle)
    return video, events, opts, n_frames / FPS


def generate(out_dir, seeds=5, scenarios=SCENARIOS):
    os.makedirs(out_dir, exist_ok=True)
    manifest, annotations = [], []
    split_for = lambda s: "test" if s == seeds - 1 else ("development" if s == seeds - 2 and seeds >= 3 else "train")
    for seed in range(seeds):
        for kind in scenarios:
            clip_id = "syn_{}_{}".format(kind, seed)
            video, events, opts, duration = render_and_track(kind, seed, out_dir, clip_id)
            manifest.append({"clip_id": clip_id, "video_path": video, "scene_id": "synthetic_scene{}".format(seed),
                             "session_id": "synthetic_session{}".format(seed), "split": split_for(seed),
                             "duration_seconds": round(duration, 2), "consent_verified": "true",
                             "viewpoint": "synthetic_elevated", "illumination": opts["illumination"],
                             "crowd_density": opts["crowd_density"], "notes": "SYNTHETIC scenario '{}' - no real people".format(kind)})
            for e in events:
                annotations.append({"clip_id": clip_id, "scene_id": "synthetic_scene{}".format(seed), "session_id": "synthetic_session{}".format(seed),
                                    "annotator": "simulator", "event_type": e["event_type"], "event_start_seconds": e["start"],
                                    "event_end_seconds": e["end"], "trigger_seconds": e["trigger"],
                                    "low_visibility": "true" if e.get("low_visibility") else "false",
                                    "visibility_reason": e.get("visibility_reason", ""), "track_id": e.get("track_id", ""),
                                    "notes": "derived from scripted motion"})
    from .dataset import MANIFEST_FIELDS, ANNOTATION_FIELDS
    for name, rows, fields in (("manifest.csv", manifest, MANIFEST_FIELDS), ("annotations.csv", annotations, ANNOTATION_FIELDS)):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(out_dir, "config.synthetic.json"), "w", encoding="utf-8") as handle:
        json.dump(SIM_CONFIG, handle, indent=2)
    return manifest, annotations


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--seeds", type=int, default=5, help="Variants per scenario; the last seed is the test split")
    p.add_argument("--scenario", action="append", choices=SCENARIOS, help="Limit to these scenarios")
    a = p.parse_args()
    manifest, annotations = generate(a.out, a.seeds, tuple(a.scenario or SCENARIOS))
    print("Wrote {} synthetic clips and {} ground-truth events to {}".format(len(manifest), len(annotations), a.out))

if __name__ == "__main__": main()
