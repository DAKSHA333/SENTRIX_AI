"""Dataset manifest/annotation templates, validation, and group-safe splitting."""
import argparse
import csv
import os
import random

MANIFEST_FIELDS = ["clip_id","video_path","scene_id","session_id","split","duration_seconds","consent_verified","viewpoint","illumination","crowd_density","notes"]
ANNOTATION_FIELDS = ["clip_id","scene_id","session_id","annotator","event_type","event_start_seconds","event_end_seconds","trigger_seconds","low_visibility","visibility_reason","track_id","notes"]


def write_template(path, fields):
    with open(path,"w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()


def init(out_dir):
    os.makedirs(out_dir,exist_ok=True)
    write_template(os.path.join(out_dir,"manifest.csv"),MANIFEST_FIELDS)
    write_template(os.path.join(out_dir,"annotations.csv"),ANNOTATION_FIELDS)


def split_groups(manifest, seed=42):
    # Connected components keep both scene and recording session together.
    parent=list(range(len(manifest)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b):
        ra,rb=root(a),root(b)
        if ra!=rb:parent[rb]=ra
    owner={}
    for i,row in enumerate(manifest):
        for key in ("scene_id","session_id"):
            value=row.get(key)
            if value:
                compound=(key,value)
                if compound in owner:join(i,owner[compound])
                else:owner[compound]=i
    groups={}
    for i,row in enumerate(manifest):groups.setdefault(root(i),[]).append(row)
    keys=list(groups);random.Random(seed).shuffle(keys)
    n=len(keys)
    if n<3:
        raise ValueError("Need at least three independent scene/session groups for train/dev/test splits")
    n_train=max(1,round(.6*n));n_dev=max(1,round(.2*n))
    if n_train+n_dev>=n:n_train=n-2;n_dev=1
    assignments={}
    for i,key in enumerate(keys):
        split="train" if i<n_train else "development" if i<n_train+n_dev else "test"
        for row in groups[key]:assignments[row["clip_id"]]=split
    return assignments


def validate(manifest, annotations, check_paths=False):
    errors=[];ids=[r.get("clip_id","") for r in manifest]
    if not manifest:errors.append("manifest is empty")
    if len(ids)!=len(set(ids)):errors.append("clip_id values are not unique")
    known=set(ids);seen_groups={}
    for row in manifest:
        cid=row.get("clip_id","")
        if not row.get("session_id") or not row.get("scene_id"):errors.append("{}: session_id and scene_id are required".format(cid))
        if row.get("consent_verified","").lower() not in ("true","yes","1"):errors.append("{}: consent_verified must be true".format(cid))
        if check_paths and row.get("video_path") and not os.path.exists(row["video_path"]):errors.append("{}: video_path does not exist".format(cid))
        split=row.get("split","")
        if split:
            for kind in ("scene_id","session_id"):
                group=row.get(kind)
                if group:
                    prior=seen_groups.setdefault((kind,group),split)
                    if prior!=split:errors.append("{} {} leaks across splits".format(kind,group))
    for row in annotations:
        cid=row.get("clip_id","")
        if cid not in known:errors.append("annotation references unknown clip_id {}".format(cid))
        try:
            start,end=float(row["event_start_seconds"]),float(row["event_end_seconds"])
            trigger=float(row.get("trigger_seconds") or start)
            if start<0 or end<start or trigger<0:raise ValueError()
            duration=float(next(m.get("duration_seconds") or 0 for m in manifest if m["clip_id"]==cid))
            if duration and end>duration:errors.append("{}: annotation exceeds clip duration".format(cid))
        except (ValueError,KeyError,StopIteration):errors.append("{}: invalid annotation times".format(cid))
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    pinit=sub.add_parser("init");pinit.add_argument("out_dir",nargs="?",default="data/benchmark")
    pval=sub.add_parser("validate");pval.add_argument("--manifest",default="data/benchmark/manifest.csv");pval.add_argument("--annotations",default="data/benchmark/annotations.csv");pval.add_argument("--check-paths",action="store_true")
    psplit=sub.add_parser("split");psplit.add_argument("--manifest",default="data/benchmark/manifest.csv");psplit.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.command=="init":init(args.out_dir);print("Wrote blank manifest and annotation templates to {}".format(args.out_dir));return
    with open(args.manifest,newline="",encoding="utf-8-sig") as f:manifest=list(csv.DictReader(f))
    if args.command=="validate":
        with open(args.annotations,newline="",encoding="utf-8-sig") as f:annotations=list(csv.DictReader(f))
        errors=validate(manifest,annotations,args.check_paths)
        if errors:
            for e in errors:print("ERROR:",e)
            raise SystemExit(1)
        print("Dataset metadata validated: {} clips, {} annotations".format(len(manifest),len(annotations)))
    else:
        assignments=split_groups(manifest,args.seed)
        for row in manifest:row["split"]=assignments[row["clip_id"]]
        with open(args.manifest,"w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=MANIFEST_FIELDS);writer.writeheader();writer.writerows(manifest)
        print("Assigned group-separated train/development/test splits in {}".format(args.manifest))

if __name__=="__main__":main()
