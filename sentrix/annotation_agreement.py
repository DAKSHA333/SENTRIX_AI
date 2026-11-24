"""Compare event annotations from two reviewers on the same held-out subset."""
import argparse
import csv
import json
import statistics
from collections import defaultdict


def agreement(rows, reviewer_a, reviewer_b, tolerance=1.0):
    grouped=defaultdict(lambda:{reviewer_a:[],reviewer_b:[]})
    for row in rows:
        reviewer=row.get("annotator","")
        if reviewer in (reviewer_a,reviewer_b):grouped[(row["clip_id"],row["event_type"])][reviewer].append(row)
    matched=total_a=total_b=0;onset_differences=[]
    for key,by in grouped.items():
        a=sorted(by[reviewer_a],key=lambda r:float(r["event_start_seconds"]))
        b=sorted(by[reviewer_b],key=lambda r:float(r["event_start_seconds"]))
        total_a+=len(a);total_b+=len(b);used=set()
        for ra in a:
            ta=float(ra["event_start_seconds"])
            choices=[(abs(ta-float(rb["event_start_seconds"])),i,rb) for i,rb in enumerate(b) if i not in used and abs(ta-float(rb["event_start_seconds"]))<=tolerance]
            if choices:
                delta,i,_=min(choices);used.add(i);matched+=1;onset_differences.append(delta)
    denom=total_a+total_b-matched
    return {"reviewer_a":reviewer_a,"reviewer_b":reviewer_b,"events_a":total_a,"events_b":total_b,
            "matched_events":matched,"event_level_f1_agreement":round(2*matched/denom,4) if denom else None,
            "median_absolute_onset_difference_seconds":round(statistics.median(onset_differences),3) if onset_differences else None,
            "tolerance_seconds":tolerance,"interpretation":"Event matching agreement only; adjudicate disagreements and report the sample size."}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("annotations_csv");p.add_argument("--reviewer-a",required=True);p.add_argument("--reviewer-b",required=True);p.add_argument("--tolerance",type=float,default=1.0);a=p.parse_args()
    with open(a.annotations_csv,newline="",encoding="utf-8-sig") as handle:rows=list(csv.DictReader(handle))
    print(json.dumps(agreement(rows,a.reviewer_a,a.reviewer_b,a.tolerance),indent=2))
if __name__=="__main__":main()
