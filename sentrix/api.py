"""Local FastAPI interface for authenticated review of alert records."""
import hmac
import json
import os
import sqlite3
from typing import Optional, Literal
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

DB_PATH=os.environ.get("SENTRIX_DB_PATH","data/sentrix.db")
REVIEWER_TOKEN=os.environ.get("SENTRIX_REVIEWER_TOKEN","")
ADMIN_TOKEN=os.environ.get("SENTRIX_ADMIN_TOKEN","")
app=FastAPI(title="SENTRIX AI Review API",version="0.1.0",description="Local human-review API for observable event alerts.")


class Decision(BaseModel):
    decision:Literal["confirmed","dismissed","escalated"]
    note:str=""


def actor(authorization:Optional[str]=Header(default=None)):
    if not REVIEWER_TOKEN and not ADMIN_TOKEN:
        raise HTTPException(status_code=503,detail="Configure reviewer/admin API tokens before enabling the review API.")
    token=authorization.removeprefix("Bearer ") if authorization else ""
    if ADMIN_TOKEN and hmac.compare_digest(token,ADMIN_TOKEN):return "admin"
    if REVIEWER_TOKEN and hmac.compare_digest(token,REVIEWER_TOKEN):return "reviewer"
    raise HTTPException(status_code=401,detail="Invalid reviewer token")


def admin_actor(authorization:Optional[str]=Header(default=None)):
    role=actor(authorization)
    if role!="admin":raise HTTPException(status_code=403,detail="Administrator token required")
    return role


def connection():
    if not os.path.isfile(DB_PATH):raise HTTPException(status_code=404,detail="Alert database not found")
    db=sqlite3.connect(DB_PATH);db.row_factory=sqlite3.Row
    columns={r[1] for r in db.execute("PRAGMA table_info(alerts)")}
    if not columns:db.close();raise HTTPException(status_code=404,detail="Alert database has not been initialized")
    if "evidence_path" not in columns:db.execute("ALTER TABLE alerts ADD COLUMN evidence_path TEXT")
    if "reviewer_note" not in columns:db.execute("ALTER TABLE alerts ADD COLUMN reviewer_note TEXT")
    if "explanation_rating" not in columns:db.execute("ALTER TABLE alerts ADD COLUMN explanation_rating INTEGER")
    db.execute("CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY, alert_id INTEGER, actor TEXT, action TEXT, note TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    db.commit()
    return db


@app.get("/health")
def health():
    return {"status":"ok","service":"SENTRIX review API"}


@app.get("/alerts")
def alerts(limit:int=100,reviewer:str=Depends(actor)):
    db=connection()
    try:rows=db.execute("SELECT id,video,event_type,timestamp_seconds,payload,decision,created_at,evidence_path,reviewer_note,explanation_rating FROM alerts ORDER BY id DESC LIMIT ?",(max(1,min(limit,500)),)).fetchall()
    finally:db.close()
    result=[]
    for row in rows:
        item=dict(row);item["payload"]=json.loads(item["payload"]);result.append(item)
    return result


@app.get("/alerts/{alert_id}")
def alert_detail(alert_id:int,reviewer:str=Depends(actor)):
    db=connection()
    try:row=db.execute("SELECT id,video,event_type,timestamp_seconds,payload,decision,created_at,evidence_path,reviewer_note,explanation_rating FROM alerts WHERE id=?",(alert_id,)).fetchone()
    finally:db.close()
    if row is None:raise HTTPException(status_code=404,detail="Alert not found")
    item=dict(row);item["payload"]=json.loads(item["payload"]);return item


@app.post("/alerts/{alert_id}/decision")
def decide(alert_id:int,body:Decision,reviewer:str=Depends(actor)):
    db=connection()
    try:
        exists=db.execute("SELECT 1 FROM alerts WHERE id=?",(alert_id,)).fetchone()
        if not exists:raise HTTPException(status_code=404,detail="Alert not found")
        db.execute("UPDATE alerts SET decision=?,reviewer_note=? WHERE id=?",(body.decision,body.note,alert_id))
        db.execute("INSERT INTO audit_log(alert_id,actor,action,note) VALUES(?,?,?,?)",(alert_id,reviewer,body.decision,body.note));db.commit()
    finally:db.close()
    return {"alert_id":alert_id,"decision":body.decision,"actor":reviewer}


@app.get("/audit")
def audit(limit:int=200,admin:str=Depends(admin_actor)):
    db=connection()
    try:rows=db.execute("SELECT id,alert_id,actor,action,note,created_at FROM audit_log ORDER BY id DESC LIMIT ?",(max(1,min(limit,1000)),)).fetchall()
    finally:db.close()
    return [dict(row) for row in rows]
