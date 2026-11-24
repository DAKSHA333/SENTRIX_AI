"""Click a zone polygon over a camera view or a recorded clip and save it to a config.

    python -m sentrix.calibrate --camera 0 --config config.demo.json
    python -m sentrix.calibrate --video clip.mp4 --frame 50 --config config.research.json --zone-name storage_door

Keys: left-click add point | u undo | r reset | s save | q quit
The view freezes on one frame for a recorded clip; for a camera press f to freeze/unfreeze.
Existing zones are drawn in grey. Saving replaces the zone with the same name, or appends it.
Click where people's FEET touch the floor: zones are tested at the bottom-centre of each person box
(zone_anchor = "bottom").
"""
import argparse
import json
import cv2
import numpy as np

WINDOW = "SENTRIX zone setup"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera", default="0", help="Camera index (default 0) or authorized stream URL")
    src.add_argument("--video", help="Recorded clip to calibrate on")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to freeze on when using --video")
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--zone-name", default="restricted_demo")
    parser.add_argument("--zone-type", default="restricted", choices=("restricted", "waiting", "entry", "storage"))
    parser.add_argument("--no-entry-alert", action="store_true", help="Save with alert_on_entry=false (e.g. a dwell-only waiting zone)")
    parser.add_argument("--dwell", action="store_true", help="Enable dwell monitoring for this zone")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    zones = config.get("restricted_zones", [])
    points = []
    source = args.video or (int(args.camera) if str(args.camera).isdigit() else args.camera)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("Could not open {}. Close other camera apps and check camera permission.".format(source))
    frozen = None
    if args.video:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frozen = cap.read()
        if not ok:
            raise SystemExit("Could not read frame {} from {}".format(args.frame, args.video))
        cap.release()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([int(x), int(y)])

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print("Click at least 3 floor points around zone '{}'. u undo, r reset, s save, q quit{}.".format(
        args.zone_name, "" if args.video else ", f freeze"))
    while True:
        if frozen is None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            frame = frozen
        preview = frame.copy()
        for zone in zones:
            if zone.get("name") != args.zone_name and len(zone.get("polygon", [])) >= 3:
                cv2.polylines(preview, [np.array(zone["polygon"], np.int32)], True, (160, 160, 160), 1)
                cv2.putText(preview, zone["name"], tuple(zone["polygon"][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        for p in points:
            cv2.circle(preview, tuple(p), 6, (0, 0, 255), -1)
        if len(points) >= 2:
            cv2.polylines(preview, [np.array(points, dtype=np.int32)], len(points) >= 3, (0, 165, 255), 2)
        cv2.putText(preview, "{} | {} pts | u undo r reset s save q quit".format(args.zone_name, len(points)), (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(WINDOW, preview)
        key = cv2.waitKey(15) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key == ord("u") and points:
            points.pop()
        elif key == ord("f") and not args.video:
            frozen = None if frozen is not None else frame.copy()
        elif key == ord("s"):
            if len(points) < 3:
                print("Click at least 3 points before saving.")
                continue
            entry = {"name": args.zone_name, "polygon": points[:], "alert_on_entry": not args.no_entry_alert,
                     "zone_type": args.zone_type, "enable_dwell": args.dwell}
            for i, zone in enumerate(zones):
                if zone.get("name") == args.zone_name:
                    zones[i] = dict(zone, **entry); break
            else:
                zones.append(entry)
            config["restricted_zones"] = zones
            config["_calibrated_frame_size"] = ("Polygon pixels are for the calibrated camera resolution: {}x{}".format(frame.shape[1], frame.shape[0]))
            with open(args.config, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
                handle.write("\n")
            print("Saved zone '{}' ({} points) to {}".format(args.zone_name, len(points), args.config))
            break
        elif key == ord("q"):
            break
    if frozen is None or not args.video:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
