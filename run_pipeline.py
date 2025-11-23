import subprocess
import sys
import time
import os
import argparse
from datetime import datetime
from db import init_db

PIPELINE_STEPS = [
    ("Data Collection", "data_collection.py"),
    ("Data Cleaning", "data_clean.py"),
    ("Sentiment Analysis", "data_sentiment.py"),
    ("Indexing to Pinecone", "store_index.py"),
]

LOG_FILE = "pipeline_log.txt"

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def run_script(script_path, retries=2):
    for attempt in range(1, retries + 1):
        log(f"Starting {script_path} (Attempt {attempt})...")
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        process.stdout.close()
        process.wait()

        if process.returncode == 0:
            log(f"{script_path} completed successfully.")
            return True
        else:
            log(f"{script_path} failed (Attempt {attempt}/{retries}). Retrying...")
            time.sleep(5)
    log(f"{script_path} failed after {retries} attempts.")
    return False

def main(include_eval=False):
    start_time = time.time()
    init_db()
    log("=" * 70)
    log("Starting Full Data Pipeline")
    log("=" * 70)

    for step_name, script in PIPELINE_STEPS:
        log(f"\nRunning Step: {step_name}")
        success = run_script(script)
        if not success:
            log(f"Step '{step_name}' encountered an error. Continuing...\n")

    if include_eval:
        log("\nRunning Evaluation Step...")
        run_script("evaluate.py")

    duration = round((time.time() - start_time) / 60, 2)
    log(f"\nPipeline finished in {duration} minutes.")
    log("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate full data pipeline.")
    parser.add_argument("--eval", action="store_true", help="Include evaluation step")
    args = parser.parse_args()
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, "w", encoding="utf-8").write("Pipeline Log Initiated\n\n")
    main(include_eval=args.eval)
