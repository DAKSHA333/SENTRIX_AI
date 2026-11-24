import json
import os
import sqlite3


class AlertStore:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path)
        try:
            os.chmod(path, 0o600)
            os.chmod(os.path.dirname(path) or ".", 0o700)
        except OSError:
            pass
        self.db.execute("CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY, video TEXT, event_type TEXT, timestamp_seconds REAL, payload TEXT, decision TEXT DEFAULT 'unreviewed', created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(alerts)")}
        if "evidence_path" not in columns:
            self.db.execute("ALTER TABLE alerts ADD COLUMN evidence_path TEXT")
        if "reviewer_note" not in columns:
            self.db.execute("ALTER TABLE alerts ADD COLUMN reviewer_note TEXT")
        if "explanation_rating" not in columns:
            self.db.execute("ALTER TABLE alerts ADD COLUMN explanation_rating INTEGER")
        self.db.execute("CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY, alert_id INTEGER, actor TEXT, action TEXT, note TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        self.db.commit()

    def add(self, video, alert):
        self.db.execute("INSERT INTO alerts(video,event_type,timestamp_seconds,payload) VALUES(?,?,?,?)", (video, alert["event_type"], alert["timestamp_seconds"], json.dumps(alert)))
        self.db.commit()
        return self.db.execute("SELECT last_insert_rowid()").fetchone()[0]

    def list(self):
        rows = self.db.execute("SELECT id,video,event_type,timestamp_seconds,payload,decision,created_at,evidence_path,reviewer_note,explanation_rating FROM alerts ORDER BY id DESC").fetchall()
        return [{"id":r[0],"video":r[1],"event_type":r[2],"timestamp_seconds":r[3],"payload":json.loads(r[4]),"decision":r[5],"created_at":r[6],"evidence_path":r[7],"reviewer_note":r[8],"explanation_rating":r[9]} for r in rows]

    def decide(self, alert_id, decision, actor="reviewer", note=""):
        if decision not in ("confirmed", "dismissed", "escalated"):
            raise ValueError("Unsupported reviewer decision")
        self.db.execute("UPDATE alerts SET decision=?, reviewer_note=? WHERE id=?", (decision, note, alert_id))
        self.db.execute("INSERT INTO audit_log(alert_id,actor,action,note) VALUES(?,?,?,?)", (alert_id, actor, decision, note))
        self.db.commit()

    def set_evidence(self, alert_id, path):
        self.db.execute("UPDATE alerts SET evidence_path=? WHERE id=?", (path, alert_id)); self.db.commit()

    def rate_explanation(self, alert_id, rating, actor="reviewer"):
        if rating not in range(1,6):raise ValueError("Rating must be 1..5")
        self.db.execute("UPDATE alerts SET explanation_rating=? WHERE id=?", (rating,alert_id))
        self.db.execute("INSERT INTO audit_log(alert_id,actor,action,note) VALUES(?,?,?,?)", (alert_id,actor,"explanation_rating",str(rating)))
        self.db.commit()

    def audit_rows(self, limit=500):
        return self.db.execute("SELECT id,alert_id,actor,action,note,created_at FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def close(self):
        self.db.close()
