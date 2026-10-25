import sqlite3
import json
import os
from datetime import datetime

def get_db_path() -> str:
    """
    Returns the configured SQLite database file path.
    Reads from DATABASE_PATH or SQLITE_PATH environment variables.
    Defaults to database/newscheck.db relative to this file's directory.
    """
    configured = os.environ.get("DATABASE_PATH") or os.environ.get("SQLITE_PATH")
    if configured:
        return os.path.abspath(configured)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "newscheck.db"))


# Backward-compatible reference
DATABASE_PATH = get_db_path()


def get_db_connection():
    db_path = get_db_path()
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            source_url TEXT,
            ml_prediction TEXT NOT NULL,
            ml_confidence REAL NOT NULL,
            ml_explanation TEXT,
            verification_status TEXT NOT NULL,
            evidence_summary TEXT NOT NULL,
            sources_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_submission(input_text, source_url, ml_prediction, ml_confidence, ml_explanation, verification_status, evidence_summary, sources):
    conn = get_db_connection()
    cursor = conn.cursor()
    sources_json = json.dumps(sources, ensure_ascii=False)
    cursor.execute("""
        INSERT INTO submissions (
            input_text, source_url, ml_prediction, ml_confidence, 
            ml_explanation, verification_status, evidence_summary, sources_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (input_text, source_url, ml_prediction, ml_confidence, ml_explanation, verification_status, evidence_summary, sources_json))
    submission_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return submission_id


def get_all_submissions(limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM submissions 
        ORDER BY created_at DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    submissions = []
    for row in rows:
        sub = dict(row)
        try:
            sub["sources"] = json.loads(sub["sources_json"]) if sub["sources_json"] else []
        except Exception:
            sub["sources"] = []
        submissions.append(sub)
    conn.close()
    return submissions


def get_submission_by_id(submission_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    sub = dict(row)
    try:
        sub["sources"] = json.loads(sub["sources_json"]) if sub["sources_json"] else []
    except Exception:
        sub["sources"] = []
    return sub


def clear_all_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM submissions")
    conn.commit()
    conn.close()


def get_dashboard_statistics():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) AS total FROM submissions")
    total_count = cursor.fetchone()["total"]
    
    cursor.execute("""
        SELECT verification_status, COUNT(*) AS count 
        FROM submissions 
        GROUP BY verification_status
    """)
    status_counts = {row["verification_status"]: row["count"] for row in cursor.fetchall()}
    
    cursor.execute("""
        SELECT ml_prediction, COUNT(*) AS count 
        FROM submissions 
        GROUP BY ml_prediction
    """)
    ml_counts = {row["ml_prediction"]: row["count"] for row in cursor.fetchall()}
    
    cursor.execute("""
        SELECT strftime('%Y-%m-%d', created_at) AS day, COUNT(*) AS count 
        FROM submissions 
        GROUP BY day 
        ORDER BY day DESC 
        LIMIT 7
    """)
    recent_activity = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    supported = sum(count for k, count in status_counts.items() if "Supported" in k)
    contradicted = sum(count for k, count in status_counts.items() if "Contradicted" in k)
    mixed = sum(count for k, count in status_counts.items() if "Mixed" in k)
    unverified = sum(count for k, count in status_counts.items() if "Unverified" in k)
    
    return {
        "total_submissions": total_count,
        "supported_count": supported,
        "contradicted_count": contradicted,
        "mixed_count": mixed,
        "unverified_count": unverified,
        "status_breakdown": status_counts,
        "ml_breakdown": ml_counts,
        "recent_activity": recent_activity
    }

