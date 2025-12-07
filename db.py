import os
import sqlite3
from datetime import datetime
from typing import Optional, List

DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "pipeline.db")

def _ensure_db_path():
    os.makedirs(DATA_DIR, exist_ok=True)

def init_db():
    """Create DB and tables if missing."""
    _ensure_db_path()
    conn = sqlite3.connect(DB_FILE, timeout=30)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS scraped_posts (
        id TEXT PRIMARY KEY,
        subreddit TEXT,
        category TEXT,
        created_utc TEXT,
        fetched_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS indexed_posts (
        id TEXT PRIMARY KEY,
        indexed_at TEXT
    )
    """)
    conn.commit()
    conn.close()

# Scraper-facing helpers
def post_exists(post_id: str) -> bool:
    """Return True if post_id already recorded in scraped_posts."""
    _ensure_db_path()
    conn = sqlite3.connect(DB_FILE, timeout=30)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM scraped_posts WHERE id = ? LIMIT 1", (post_id,))
    r = cur.fetchone()
    conn.close()
    return r is not None

def mark_scraped(post_id: str, subreddit: str, category: str, created_utc_iso: str):
    """Insert or update scraped_posts for given id."""
    _ensure_db_path()
    conn = sqlite3.connect(DB_FILE, timeout=30)
    cur = conn.cursor()
    fetched_at = datetime.utcnow().isoformat()
    try:
        cur.execute("""
            INSERT OR REPLACE INTO scraped_posts (id, subreddit, category, created_utc, fetched_at)
            VALUES (?, ?, ?, ?, ?)
        """, (post_id, subreddit, category, created_utc_iso, fetched_at))
        conn.commit()
    finally:
        conn.close()

def get_last_time_for_subreddit(subreddit: str) -> Optional[str]:
    """Return latest created_utc (ISO string) recorded for subreddit, or None."""
    _ensure_db_path()
    conn = sqlite3.connect(DB_FILE, timeout=30)
    cur = conn.cursor()
    cur.execute("""
        SELECT created_utc FROM scraped_posts
        WHERE subreddit = ?
        ORDER BY datetime(created_utc) DESC
        LIMIT 1
    """, (subreddit,))
    r = cur.fetchone()
    conn.close()
    return r[0] if r else None

# Indexer-facing helpers
def get_unindexed_ids(limit: Optional[int] = None) -> List[str]:
    """
    Return list of scraped post ids that are NOT yet indexed.
    Optionally limit the number returned.
    """
    _ensure_db_path()
    conn = sqlite3.connect(DB_FILE, timeout=30)
    cur = conn.cursor()
    q = """
        SELECT s.id FROM scraped_posts s
        LEFT JOIN indexed_posts i ON s.id = i.id
        WHERE i.id IS NULL
        ORDER BY datetime(s.created_utc) ASC
    """
    if limit:
        q = q + " LIMIT ?"
        cur.execute(q, (limit,))
    else:
        cur.execute(q)
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

def mark_indexed(post_id: str):
    """Mark post id as indexed (insert into indexed_posts)."""
    _ensure_db_path()
    conn = sqlite3.connect(DB_FILE, timeout=30)
    cur = conn.cursor()
    indexed_at = datetime.utcnow().isoformat()
    try:
        cur.execute("""
            INSERT OR IGNORE INTO indexed_posts (id, indexed_at)
            VALUES (?, ?)
        """, (post_id, indexed_at))
        conn.commit()
    finally:
        conn.close()

# Small utility for debug
def count_unindexed() -> int:
    _ensure_db_path()
    conn = sqlite3.connect(DB_FILE, timeout=30)
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM scraped_posts s
        LEFT JOIN indexed_posts i ON s.id = i.id
        WHERE i.id IS NULL
    """)
    r = cur.fetchone()
    conn.close()
    return int(r[0] or 0)
