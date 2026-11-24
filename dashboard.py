"""Local reviewer dashboard with role-aware decisions and an audit trail."""
import hmac
import json
import os
import sqlite3
import streamlit as st
import pandas as pd

DB_PATH = os.environ.get("SENTRIX_DB_PATH", "data/sentrix.db")
REVIEWER_PASSWORD = os.environ.get("SENTRIX_REVIEWER_PASSWORD", "")
ADMIN_PASSWORD = os.environ.get("SENTRIX_ADMIN_PASSWORD", "")
st.set_page_config(page_title="SENTRIX AI Review", layout="wide")
st.title("SENTRIX AI · Alert review")
st.caption("Alerts describe configured visual cues. A human reviewer makes every decision.")

if REVIEWER_PASSWORD or ADMIN_PASSWORD:
    if "sentrix_role" not in st.session_state:
        st.session_state.sentrix_role = None
    if st.session_state.sentrix_role is None:
        with st.form("login"):
            password = st.text_input("Reviewer or administrator password", type="password")
            submit = st.form_submit_button("Sign in")
        if submit:
            if ADMIN_PASSWORD and hmac.compare_digest(password, ADMIN_PASSWORD):
                st.session_state.sentrix_role = "admin"
            elif REVIEWER_PASSWORD and hmac.compare_digest(password, REVIEWER_PASSWORD):
                st.session_state.sentrix_role = "reviewer"
            else:
                st.error("Password not recognized.")
        if st.session_state.sentrix_role is None:
            st.stop()
    role = st.session_state.sentrix_role
else:
    role = "local_demo"
    st.warning("Authentication is not configured. Use localhost for a controlled demo only; set SENTRIX_REVIEWER_PASSWORD and SENTRIX_ADMIN_PASSWORD before shared use.")

if st.button("Sign out"):
    st.session_state.sentrix_role = None
    st.rerun()

db = None
try:
    db = sqlite3.connect(DB_PATH)
    columns={row[1] for row in db.execute("PRAGMA table_info(alerts)")}
    if "evidence_path" not in columns:db.execute("ALTER TABLE alerts ADD COLUMN evidence_path TEXT")
    if "reviewer_note" not in columns:db.execute("ALTER TABLE alerts ADD COLUMN reviewer_note TEXT")
    if "explanation_rating" not in columns:db.execute("ALTER TABLE alerts ADD COLUMN explanation_rating INTEGER")
    db.execute("CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY, alert_id INTEGER, actor TEXT, action TEXT, note TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    db.commit()
    rows = db.execute("SELECT id, video, event_type, timestamp_seconds, payload, decision, created_at, evidence_path, reviewer_note, explanation_rating FROM alerts ORDER BY id DESC").fetchall()
except sqlite3.OperationalError:
    rows = []

if not rows:
    st.info("No alerts in {} yet. Run the video or camera pipeline first (or `python -m sentrix.demo_synthetic` and point SENTRIX_DB_PATH at reports/synthetic/runs/context.db).".format(DB_PATH))
    if db is not None:
        db.close()
else:
    data = pd.DataFrame(rows, columns=["id", "source", "event_type", "timestamp_seconds", "payload", "decision", "created_at", "evidence_path", "reviewer_note", "explanation_rating"])
    f1, f2, f3 = st.columns(3)
    with f1: event_filter = st.selectbox("Event", ["All"] + sorted(data.event_type.unique().tolist()))
    with f2: decision_filter = st.selectbox("Review status", ["All", "unreviewed", "confirmed", "dismissed", "escalated"])
    with f3: quality_filter = st.selectbox("Evidence quality", ["All", "SUFFICIENT", "INSUFFICIENT_EVIDENCE"])
    visible = data.copy()
    if event_filter != "All": visible = visible[visible.event_type == event_filter]
    if decision_filter != "All": visible = visible[visible.decision == decision_filter]
    if quality_filter != "All":
        visible = visible[visible.payload.map(lambda x: json.loads(x).get("evidence_quality") == quality_filter)]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Alerts shown", len(visible))
    m2.metric("Unreviewed", int((visible.decision == "unreviewed").sum()))
    m3.metric("Confirmed", int((visible.decision == "confirmed").sum()))
    m4.metric("Dismissed", int((visible.decision == "dismissed").sum()))
    m5.metric("Withheld (low evidence)", int((visible.event_type == "insufficient_evidence").sum()))
    if len(visible):
        by_type = visible.assign(shown_type=visible.payload.map(lambda x: json.loads(x).get("candidate_event_type") or json.loads(x).get("event_type")))
        st.bar_chart(by_type.groupby(["shown_type", "decision"]).size().unstack(fill_value=0))

    for row in visible.itertuples(index=False):
        payload = json.loads(row.payload)
        label = "{} · {} · {}s · {}".format(row.event_type.replace("_", " "), row.source, round(row.timestamp_seconds, 1), row.decision)
        with st.expander(label):
            st.write(payload.get("reason", ""))
            if payload.get("evidence_reasons"):
                st.warning("Evidence-quality flags: " + ", ".join(r.replace("_", " ").lower() for r in payload["evidence_reasons"]))
            st.json({"candidate_event_type": payload.get("candidate_event_type"), "zone": payload.get("zone"),
                     "evidence_quality": payload.get("evidence_quality"), "reason_code": payload.get("reason_code"),
                     "evidence_reasons": payload.get("evidence_reasons"), "method": payload.get("method"),
                     "measurements": payload.get("measurements")})
            if row.evidence_path and os.path.exists(row.evidence_path):
                st.video(row.evidence_path)
            elif row.evidence_path:
                st.caption("Evidence clip is unavailable at its recorded path.")
            if role in ("reviewer", "admin", "local_demo"):
                st.caption("Explanation rating: 1 = does not match evidence, 5 = clearly matches evidence")
                old_rating = int(row.explanation_rating) if pd.notna(row.explanation_rating) else 3
                rating = st.select_slider("How well does the reason match the visible evidence?", options=[1,2,3,4,5], value=old_rating, key="rating_{}".format(row.id))
                if st.button("Save explanation rating", key="rating_save_{}".format(row.id)):
                    db.execute("UPDATE alerts SET explanation_rating=? WHERE id=?", (rating,row.id))
                    db.execute("INSERT INTO audit_log(alert_id,actor,action,note) VALUES(?,?,?,?)", (row.id,role,"explanation_rating",str(rating)))
                    db.commit();st.rerun()
                note = st.text_input("Reviewer note", value=row.reviewer_note or "", key="note_{}".format(row.id))
                cols = st.columns(3)
                for col, decision in zip(cols, ("confirmed", "dismissed", "escalated")):
                    if col.button(decision.title(), key="decision_{}_{}".format(row.id, decision)):
                        db.execute("UPDATE alerts SET decision=?, reviewer_note=? WHERE id=?", (decision, note, row.id))
                        db.execute("INSERT INTO audit_log(alert_id,actor,action,note) VALUES(?,?,?,?)", (row.id, role, decision, note))
                        db.commit(); st.rerun()
    st.subheader("Alert table")
    st.dataframe(visible.drop(columns=["payload"]), use_container_width=True, hide_index=True)

    if role == "admin":
        st.subheader("Audit history")
        audit = db.execute("SELECT id,alert_id,actor,action,note,created_at FROM audit_log ORDER BY id DESC LIMIT 500").fetchall()
        st.dataframe(pd.DataFrame(audit, columns=["id","alert_id","actor","action","note","created_at"]), use_container_width=True, hide_index=True)
    db.close()
