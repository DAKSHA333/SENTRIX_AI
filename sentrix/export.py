"""Export local alerts to a flat CSV for reproducible evaluation."""
import argparse
import csv
import json
import os
import sqlite3

FIELDS = ["clip_id","event_type","timestamp_seconds","evidence_quality","candidate_event_type","reason_code","evidence_reasons","decision","method","explanation_rating"]


def export_db(database, out):
    db = sqlite3.connect(database)
    columns={row[1] for row in db.execute("PRAGMA table_info(alerts)")}
    rating_col="explanation_rating" if "explanation_rating" in columns else "NULL"
    rows = db.execute("SELECT video,event_type,timestamp_seconds,payload,decision,{} FROM alerts ORDER BY id".format(rating_col)).fetchall()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for video,event,time_s,payload,decision,rating in rows:
            body = json.loads(payload)
            writer.writerow({"clip_id":video,"event_type":event,"timestamp_seconds":time_s,
                "evidence_quality":body.get("evidence_quality",""),"candidate_event_type":body.get("candidate_event_type",""),
                "reason_code":body.get("reason_code",""),"evidence_reasons":";".join(body.get("evidence_reasons") or []),
                "decision":decision,"method":body.get("method",""),"explanation_rating":rating})
    db.close()
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/sentrix.db")
    parser.add_argument("--out", default="data/predictions.csv")
    args = parser.parse_args()
    n = export_db(args.database, args.out)
    print("Exported {} alert rows to {}".format(n, args.out))

if __name__ == "__main__": main()
