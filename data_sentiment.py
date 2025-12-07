import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Always use SEMANTIC-CLEAN file as input
INPUT_FILE = DATA_DIR / "reddit_data_semantic_clean.csv"
OUTPUT_FILE = DATA_DIR / "reddit_data_sentiment.csv"
LOG_FILE = DATA_DIR / "sentiment_log.txt"


def log(msg):
    ts = datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    log("Starting sentiment analysis (semantic dataset only)")

    if not INPUT_FILE.exists():
        log(f"ERROR: Missing semantic-clean file: {INPUT_FILE}")
        sys.exit(1)

    log(f"Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")
    log(f"Loaded {len(df)} rows")

    # Ensure NLTK resources
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    sentiments = []
    for text in df["clean_text"].astype(str):
        scores = sia.polarity_scores(text)
        if scores["compound"] >= 0.2:
            sentiments.append("positive")
        elif scores["compound"] <= -0.2:
            sentiments.append("negative")
        else:
            sentiments.append("neutral")

    df["sentiment"] = sentiments

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    log(f"Saved sentiment file: {OUTPUT_FILE} ({len(df)} rows)")

    log("Sentiment analysis complete.")
    log("=" * 60)


if __name__ == "__main__":
    main()
