import pandas as pd
import re
import json
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

# =============================
# CONFIGURATION
# =============================
DATA_DIR = "data"
CONFIG_PATH = "config/subreddits.json"
RAW_FILE = os.path.join(DATA_DIR, "reddit_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "reddit_data_clean.csv")
LOG_FILE = os.path.join(DATA_DIR, "data_clean_log.txt")

# -----------------------------
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# -----------------------------
def load_subreddits():
    if not os.path.exists(CONFIG_PATH):
        log("Subreddit config file not found.")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def find_keywords(text, keywords):
    found = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text.lower())]
    return found

# -----------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_FILE):
        log(f"Missing input data: {RAW_FILE}")
        sys.exit(1)

    df = pd.read_csv(RAW_FILE)
    log(f"Loaded dataset with {len(df)} rows.")

    # Drop duplicates
    df = df.drop_duplicates(subset=["id"], keep="last")
    df = df.drop_duplicates(subset=["title", "content"], keep="last")
    df["full_text"] = df[["title", "content"]].astype(str).agg(" ".join, axis=1)

    # Load subreddit categories for keyword mapping
    subreddit_config = load_subreddits()
    categories = list(subreddit_config.keys())

    # Software dictionary (core domain list)
    keywords_dict = {
        "Law": ["clio", "filevine", "smokeball", "practicepanther", "lexisnexis",
                "westlaw", "imanage", "everlaw", "relativity", "lawpay", "ediscovery"],
        "Construction": ["autocad", "revit", "bim", "sketchup", "bluebeam", "procore",
                         "plangrid", "primavera", "estimating software"],
        "Tech": ["jira", "docker", "kubernetes", "ansible", "aws", "azure", "gcp",
                 "firewall", "splunk", "linux", "servicenow", "active directory"]
    }

    df["keywords_found"] = [[] for _ in range(len(df))]
    for category, keywords in keywords_dict.items():
        mask = df["category"] == category
        df.loc[mask, "keywords_found"] = df.loc[mask].apply(
            lambda row: list(set(row["keywords_found"]) | set(find_keywords(str(row["full_text"]), keywords))),
            axis=1
        )

    df["software_flag"] = df["keywords_found"].apply(lambda x: len(x) > 0)
    df_clean = df[df["software_flag"] == True].copy()

    df_clean["clean_text"] = df_clean["full_text"].apply(clean_text)
    df_clean.to_csv(OUTPUT_FILE, index=False)
    log(f"Cleaned dataset saved with {len(df_clean)} rows to {OUTPUT_FILE}.")

    # Keyword summary (optional logging)
    all_keywords = df_clean["keywords_found"].explode()
    keyword_counts = all_keywords.value_counts().head(10)
    log("Top keywords found:\n" + keyword_counts.to_string())

    log("Data cleaning completed successfully.")

if __name__ == "__main__":
    main()
