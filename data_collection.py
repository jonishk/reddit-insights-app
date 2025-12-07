# data_collection.py (bulletproof Windows-safe version)
import os
import sys
import time
import json
import pandas as pd
from datetime import datetime
import praw

# Unified DB helpers
from db import init_db, post_exists, mark_scraped, get_last_time_for_subreddit


# SAFE LOGGING
def sanitize(s):
    """Remove unsupported characters for Windows console."""
    try:
        # Try printing; if fails, strip to ASCII
        s.encode(sys.stdout.encoding or "utf-8", errors="strict")
        return s
    except:
        return s.encode("ascii", errors="ignore").decode("ascii")


def log(message):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"

    # Print sanitized line to console
    print(sanitize(line), flush=True)

    # Write full UTF-8 to log file
    with open("data/scrape_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# CONFIG
CONFIG_PATH = "config/subreddits.json"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "reddit_data.csv")

# Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID", "qL6M97vuwERUMPfq3f53XQ"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET", "XsJsBSfZj3jZNccXiJG77CtDanDojg"),
    user_agent=os.getenv("REDDIT_USER_AGENT", "RedditScraper-MSDS696"),
)

POST_LIMIT = 100
COMMENT_LIMIT = 15
MAX_RETRIES = 3
SLEEP = 3

# Helpers
def safe_str(x):
    try:
        return str(x) if x is not None else ""
    except:
        return ""


def load_subreddits():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing():
    if os.path.exists(OUTPUT_FILE):
        return pd.read_csv(OUTPUT_FILE, parse_dates=["created_utc"])
    return pd.DataFrame(columns=[
        "id", "category", "subreddit", "title", "content", "author",
        "score", "num_comments", "created_utc", "edited", "type", "parent_id"
    ])

# Scraper
def fetch_subreddit(subreddit_name, category, last_time):
    data = []
    post_count = 0
    comment_count = 0

    try:
        subreddit = reddit.subreddit(subreddit_name)
    except Exception as e:
        log(f"ERROR accessing r/{subreddit_name}: {e}")
        return data, 0, 0

    for post in subreddit.new(limit=POST_LIMIT):
        try:
            post_time = datetime.utcfromtimestamp(post.created_utc)
        except:
            continue

        if last_time and post_time <= last_time:
            continue
        if post_exists(post.id):
            continue

        # Collect post
        data.append({
            "id": post.id,
            "category": category,
            "subreddit": subreddit_name,
            "title": safe_str(post.title),
            "content": safe_str(post.selftext),
            "author": safe_str(post.author),
            "score": post.score,
            "num_comments": post.num_comments,
            "created_utc": post_time,
            "edited": post.edited if post.edited else False,
            "type": "post",
            "parent_id": None
        })
        mark_scraped(post.id, subreddit_name, category, post_time.isoformat())
        post_count += 1

        # Comments
        try:
            post.comments.replace_more(limit=0)
            for c in post.comments.list()[:COMMENT_LIMIT]:
                try:
                    ct = datetime.utcfromtimestamp(c.created_utc)
                except:
                    continue

                if last_time and ct <= last_time:
                    continue
                if post_exists(c.id):
                    continue

                data.append({
                    "id": c.id,
                    "category": category,
                    "subreddit": subreddit_name,
                    "title": None,
                    "content": safe_str(c.body),
                    "author": safe_str(c.author),
                    "score": c.score,
                    "num_comments": None,
                    "created_utc": ct,
                    "edited": c.edited if c.edited else False,
                    "type": "comment",
                    "parent_id": safe_str(c.parent_id)
                })
                mark_scraped(c.id, subreddit_name, category, ct.isoformat())
                comment_count += 1

        except Exception as e:
            log(f"WARNING comments r/{subreddit_name}: {e}")

    return data, post_count, comment_count


# Main
def main():
    init_db()
    subs = load_subreddits()
    existing = load_existing()

    last_times_csv = existing.groupby("subreddit")["created_utc"].max().to_dict()

    all_rows = []
    total_posts = 0
    total_comments = 0

    log("Starting Reddit scraper…")

    for category, sub_list in subs.items():
        for sub in sub_list:
            retries = 0
            success = False

            while retries < MAX_RETRIES and not success:
                try:
                    log(f"Fetching r/{sub} ({category}) attempt {retries+1}")

                    db_time = get_last_time_for_subreddit(sub)
                    last_time = None
                    if db_time:
                        try:
                            last_time = datetime.fromisoformat(db_time)
                        except:
                            pass
                    if not last_time and sub in last_times_csv:
                        last_time = last_times_csv[sub]

                    rows, p, c = fetch_subreddit(sub, category, last_time)
                    all_rows.extend(rows)
                    total_posts += p
                    total_comments += c

                    if p or c:
                        log(f"→ r/{sub}: {p} posts, {c} comments")
                    else:
                        log(f"No new items for r/{sub}")

                    success = True
                except Exception as e:
                    retries += 1
                    log(f"ERROR r/{sub}: {e} — retry {retries}/{MAX_RETRIES}")
                    time.sleep(3)

    if not all_rows:
        log("No new data.")
        return

    df_new = pd.DataFrame(all_rows)
    df = pd.concat([existing, df_new], ignore_index=True).drop_duplicates(subset=["id"])
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    log(f"### DONE: {total_posts} posts + {total_comments} comments added.")
    log(f"Total records in dataset: {len(df)}")


if __name__ == "__main__":
    main()
