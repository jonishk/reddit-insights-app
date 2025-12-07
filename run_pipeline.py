
"""
Run full pipeline in order:
  1) data_collection.py
  2) data_clean_and_classify.py
  3) data_sentiment.py
  4) store_index.py
"""
import subprocess
import sys
import time
import os
import argparse
from datetime import datetime
from pathlib import Path

from db import init_db

PIPELINE_STEPS = [
    ("Data Collection", "data_collection.py"),
    ("Clean + Semantic", "data_clean_and_classify.py"),
    ("Sentiment", "data_sentiment.py"),
    ("Indexing", "store_index.py"),
]

LOG_FILE = "logs/pipeline_log.txt"

def log(msg):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_script(script, env=None):
    log(f"Starting script: {script}")
    p = subprocess.Popen(
        [sys.executable, script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    for line in p.stdout:
        print(line, end="")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    p.wait()
    if p.returncode != 0:
        log(f"ERROR running {script} (exit {p.returncode})")
    else:
        log(f"Completed: {script}")
    return p.returncode == 0

def main(include_eval=False):
    init_db()
    start = time.time()
    log("=" * 80)
    log("STARTING FULL PIPELINE")
    log("=" * 80)

    for name, script in PIPELINE_STEPS:
        log(f"--- Running Step: {name} ---")
        env = os.environ.copy()
        if name == "Sentiment":
            # tell sentiment to read semantic-clean file
            env["SENTIMENT_INPUT_FILE"] = "data/reddit_data_semantic_clean.csv"
            log("Injected SENTIMENT_INPUT_FILE=data/reddit_data_semantic_clean.csv")
        ok = run_script(script, env)
        if not ok:
            log(f"Step failed: {name} (continuing...)")

    if include_eval:
        run_script("evaluate.py")

    mins = round((time.time() - start) / 60, 2)
    log(f"Pipeline finished in {mins} minutes.")
    log("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, "w", encoding="utf-8").write("Pipeline Log Initiated\n\n")
    main(include_eval=args.eval)
