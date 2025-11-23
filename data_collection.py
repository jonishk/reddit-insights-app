import praw
import pandas as pd
import os
import time
import json
import sqlite3
from datetime import datetime
import sys

# =============================
# ENV + DIRECTORIES
# =============================
sys.stdout.reconfigure(line_buffering=True)

CONFIG_PATH = os.getenv("SUBREDDIT_CONFIG", "config/subreddits.json")

DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "reddit_data.csv")
LOG_FILE = os.path.join(DATA_DIR, "scrape_log.txt")
DB_FILE = os.path.join(DATA_DIR, "scrape_tracker.db")

# =============================
# REDDIT API (Render-ready)
# =============================
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT", "RedditScraper-MSDS696")
)

POST_LIMIT = int(os.getenv("POST_LIMIT", 100))
COMMENT_LIMIT = int(os.getenv("COMMENT_LIMIT", 15))
SLEEP_TIME = int(os.getenv("SCRAPE_SLEEP", 8))
MAX_RETRIES = int(os.getenv("SCRAPE_RETRIES", 3))

# =============================
# DATABASE HELPERS
# =============================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scraped_ids (
            id TEXT PRIMARY KEY,
            subreddit TEXT,
            category TEXT,
            created_utc TEXT
        )
    """)
    conn.commit()
    conn.close()

def already_scraped(post_id):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM scraped_ids WHERE id = ?", (post_id,))
    result = cur.fetchone()
    conn.close()
    return result is not None

def record_scraped(post_id, subreddit, category, created_utc):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO scraped_ids (id, subreddit, category, created_utc) VALUES (?, ?, ?, ?)",
        (post_id, subreddit, category, created_utc)
    )
    conn.commit()
    conn.close()

# =============================
# LOGGING
# =============================
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# =============================
# LOADERS
# =============================
def load_subreddits():
    if not os.path.exists(CONFIG_PATH):
        log("ERROR: config/subreddits.json file not found.")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_existing_data():
    if os.path.exists(OUTPUT_FILE):
        return pd.read_csv(OUTPUT_FILE, parse_dates=["created_utc"])
    return pd.DataFrame(columns=[
        "id", "category", "subreddit", "title", "content", "author",
        "score", "num_comments", "created_utc", "edited", "type", "parent_id"
    ])

# =============================
# SCRAPER
# =============================
def fetch_subreddit_posts(subreddit_name, category, last_time=None):
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []

    for post in subreddit.new(limit=POST_LIMIT):

        post_time = datetime.utcfromtimestamp(post.created_utc)

        if (last_time and post_time <= last_time) or already_scraped(post.id):
            continue

        posts_data.append({
            "id": post.id,
            "category": category,
            "subreddit": subreddit_name,
            "title": post.title,
            "content": post.selftext,
            "author": str(post.author),
            "score": post.score,
            "num_comments": post.num_comments,
            "created_utc": post_time,
            "edited": post.edited if post.edited else False,
            "type": "post",
            "parent_id": None
        })

        record_scraped(post.id, subreddit_name, category, post_time.isoformat())

        # comments
        try:
            post.comments.replace_more(limit=0)
            for comment in post.comments.list()[:COMMENT_LIMIT]:

                comment_time = datetime.utcfromtimestamp(comment.created_utc)

                if (last_time and comment_time <= last_time) or already_scraped(comment.id):
                    continue

                posts_data.append({
                    "id": comment.id,
                    "category": category,
                    "subreddit": subreddit_name,
                    "title": None,
                    "content": comment.body,
                    "author": str(comment.author),
                    "score": comment.score,
                    "num_comments": None,
                    "created_utc": comment_time,
                    "edited": comment.edited if comment.edited else False,
                    "type": "comment",
                    "parent_id": comment.parent_id
                })

                record_scraped(comment.id, subreddit_name, category, comment_time.isoformat())

        except Exception as e:
            log(f"Error fetching comments for r/{subreddit_name}: {e}")

    return posts_data

# =============================
# MAIN
# =============================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    init_db()

    subreddits_config = load_subreddits()
    df_existing = load_existing_data()

    last_times = (
        df_existing.groupby("subreddit")["created_utc"].max().to_dict()
        if not df_existing.empty else {}
    )

    all_data = []
    log("Starting Reddit scraper (Render-compatible)...")

    for category, subreddits in subreddits_config.items():
        for sub in subreddits:
            attempt = 0
            success = False

            while attempt < MAX_RETRIES and not success:
                try:
                    log(f"Fetching r/{sub} ({category}) attempt {attempt+1}")
                    last_time = last_times.get(sub)
                    posts = fetch_subreddit_posts(sub, category, last_time)

                    if posts:
                        all_data.extend(posts)
                        log(f"→ {len(posts)} new items from r/{sub}")
                    else:
                        log(f"No new posts for r/{sub}")

                    success = True
                    time.sleep(SLEEP_TIME)

                except Exception as e:
                    attempt += 1
                    log(f"Error: {e} — retrying...")
                    time.sleep(10)

            if not success:
                log(f"Failed to fetch r/{sub} after retries.")

    if not all_data:
        log("No new data. Everything is up to date.")
        return

    df_new = pd.DataFrame(all_data)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["id"], keep="last")
    df_combined.to_csv(OUTPUT_FILE, index=False)

    log(f"Saved {len(df_new)} new rows. Total: {len(df_combined)}")
    log("Scraper finished.")

if __name__ == "__main__":
    main()
