"""
Structured call log. SQLite stands in for Redshift at this project's scale.
WAL mode is mandatory here, not optional -- without it, a reader (the REST
API) and a writer (the pipeline) hitting this file concurrently will throw
"database is locked" under any real load.
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "logs" / "calls.db"

# Cold-start guard: below this many historical calls, percentile-based
# flagging is statistical noise, so fall back to a fixed threshold.
MIN_CALLS_FOR_PERCENTILE = 20
FALLBACK_THRESHOLD_MS = 6000.0
FLAG_PERCENTILE = 95


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                call_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                transcript TEXT,
                reply TEXT,
                vad_ms REAL,
                stt_ms REAL,
                llm_ttft_ms REAL,
                llm_total_ms REAL,
                tts_ms REAL,
                total_ms REAL,
                flagged INTEGER NOT NULL DEFAULT 0
            )
        """)


def _flag_threshold(conn) -> float:
    """Returns the total_ms value above which a call is 'flagged'."""
    row = conn.execute("SELECT COUNT(*) AS n FROM calls").fetchone()
    if row["n"] < MIN_CALLS_FOR_PERCENTILE:
        return FALLBACK_THRESHOLD_MS
    totals = [r["total_ms"] for r in conn.execute("SELECT total_ms FROM calls")]
    totals.sort()
    idx = min(int(len(totals) * FLAG_PERCENTILE / 100), len(totals) - 1)
    return totals[idx]


def insert_call(call_id: str, transcript: str, reply: str, latency: dict):
    """latency: dict with keys vad, stt, llm_ttft, llm_total, tts, total (ms)."""
    with _connect() as conn:
        threshold = _flag_threshold(conn)
        flagged = int(latency["total"] > threshold)
        conn.execute(
            """INSERT INTO calls
               (call_id, timestamp, transcript, reply, vad_ms, stt_ms,
                llm_ttft_ms, llm_total_ms, tts_ms, total_ms, flagged)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (call_id, time.time(), transcript, reply,
             latency["vad"], latency["stt"], latency["llm_ttft"],
             latency["llm_total"], latency["tts"], latency["total"], flagged),
        )


def get_call(call_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        return dict(row) if row else None


def get_flagged_calls(since: float | None = None) -> list[dict]:
    with _connect() as conn:
        if since is not None:
            rows = conn.execute(
                "SELECT * FROM calls WHERE flagged = 1 AND timestamp > ? ORDER BY timestamp DESC",
                (since,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM calls WHERE flagged = 1 ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]


def get_metrics() -> dict:
    """Summary stats for the dashboard: counts, latency percentiles, flag rate."""
    with _connect() as conn:
        rows = conn.execute("SELECT total_ms, flagged FROM calls ORDER BY timestamp").fetchall()
    if not rows:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "flagged_rate": 0.0}

    totals = sorted(r["total_ms"] for r in rows)
    n = len(totals)

    def pct(p):
        return totals[min(int(n * p / 100), n - 1)]

    flagged_count = sum(r["flagged"] for r in rows)
    return {
        "count": n,
        "p50_ms": round(pct(50), 1),
        "p95_ms": round(pct(95), 1),
        "p99_ms": round(pct(99), 1),
        "flagged_rate": round(flagged_count / n, 3),
    }
