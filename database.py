"""database.py - SQLite storage for prediction history (Python standard library only)."""

import sqlite3
from datetime import datetime

DB_FILE = "history.db"


def _connect():
    return sqlite3.connect(DB_FILE)


def init_db():
    with _connect() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   created_at TEXT NOT NULL,
                   source TEXT NOT NULL,
                   preview TEXT NOT NULL,
                   label TEXT NOT NULL,
                   confidence REAL NOT NULL)"""
        )


def save_prediction(source, text, label, confidence):
    preview = text.replace("\n", " ").strip()
    preview = preview[:110] + ("..." if len(preview) > 110 else "")
    with _connect() as con:
        con.execute(
            "INSERT INTO predictions (created_at, source, preview, label, confidence) VALUES (?,?,?,?,?)",
            (datetime.now().strftime("%d %b %Y, %H:%M"), source, preview, label, confidence),
        )


def recent(limit=50):
    with _connect() as con:
        rows = con.execute(
            "SELECT created_at, source, preview, label, confidence FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(zip(("time", "source", "preview", "label", "confidence"), r)) for r in rows]


def stats():
    with _connect() as con:
        total = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        fake = con.execute("SELECT COUNT(*) FROM predictions WHERE label='FAKE'").fetchone()[0]
    return {"total": total, "fake": fake, "real": total - fake}


def clear():
    with _connect() as con:
        con.execute("DELETE FROM predictions")