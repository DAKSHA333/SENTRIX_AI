"""Compute scene-local track ID switches if frame-level GT IDs are annotated."""
import argparse
import csv
import json
from collections import defaultdict


def id_switches(rows):
    by_track=defaultdict(list)
    matched=0
    for row in rows:
        gt=row.get("gt_id","");pred=row.get("predicted_id","")
        if gt in ("",None) or pred in ("",None):continue
        matched+=1;by_track[(row["clip_id"],gt)].append((int(row["frame_index"]),str(pred)))
    switches=0
    for sequence in by_track.values():
        sequence.sort()
        previous=None
        for _,ident in sequence:
            if previous is not None and ident!=previous:switches+=1
            previous=ident
    return {"matched_track_frames":matched,"ground_truth_tracks":len(by_track),"id_switches":switches,
            "switches_per_1000_matched_frames":round(switches*1000/matched,3) if matched else None}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("annotations_csv");a=p.parse_args()
    with open(a.annotations_csv,newline="",encoding="utf-8-sig") as f:rows=list(csv.DictReader(f))
    print(json.dumps(id_switches(rows),indent=2))
if __name__=="__main__":main()
