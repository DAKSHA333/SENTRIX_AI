"""Delete expired evidence clips and their alert records according to retention policy."""
import argparse
import os
import sqlite3
import time
from .config import Settings


def purge(database, evidence_dir, days, dry_run=False, trajectory_csv=None):
    cutoff=time.time()-days*86400
    if not os.path.exists(database):
        return {"database_exists":False,"expired_alerts":0,"evidence_files_removed":0,"trajectory_file_removed":False,"dry_run":dry_run,"retention_days":days}
    db=sqlite3.connect(database)
    rows=db.execute("SELECT id,evidence_path,created_at FROM alerts WHERE created_at < datetime('now', ?) ",("-{} days".format(days),)).fetchall()
    removed=0
    for ident,path,created in rows:
        if path and os.path.isfile(path):
            if not dry_run:os.remove(path)
            removed+=1
        if not dry_run:
            db.execute("DELETE FROM alerts WHERE id=?",(ident,))
            db.execute("DELETE FROM audit_log WHERE alert_id=?",(ident,))
    if not dry_run:db.commit()
    db.close()
    features_removed=False
    if trajectory_csv and os.path.isfile(trajectory_csv) and os.path.getmtime(trajectory_csv)<cutoff:
        if not dry_run:os.remove(trajectory_csv)
        features_removed=True
    return {"expired_alerts":len(rows),"evidence_files_removed":removed,"trajectory_file_removed":features_removed,"dry_run":dry_run,"retention_days":days}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",default="config.example.json");p.add_argument("--dry-run",action="store_true")
    a=p.parse_args();s=Settings.from_json(a.config);print(purge(s.database,s.evidence_dir,s.retention_days,a.dry_run,s.trajectory_csv))
if __name__=="__main__":main()
