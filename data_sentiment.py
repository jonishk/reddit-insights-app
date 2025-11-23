import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import word_tokenize, bigrams
from collections import Counter
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

# =============================
# CONFIGURATION
# =============================
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "reddit_data_clean.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "reddit_data_sentiment.csv")
LOG_FILE = os.path.join(DATA_DIR, "sentiment_log.txt")

# -----------------------------
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# -----------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(INPUT_FILE):
        log(f"Missing cleaned dataset: {INPUT_FILE}")
        sys.exit(1)

    log("Loading NLTK components...")
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    stop_words = set(stopwords.words("english"))
    sia = SentimentIntensityAnalyzer()

    df = pd.read_csv(INPUT_FILE)
    log(f"Loaded cleaned dataset with {len(df)} rows.")

    def clean_nan_words(text):
        text = str(text).lower()
        return text.replace("nan", "").strip()

    df["clean_text"] = df["clean_text"].apply(clean_nan_words)

    def get_sentiment(text):
        score = sia.polarity_scores(str(text))["compound"]
        if score > 0.05:
            return "positive"
        elif score < -0.05:
            return "negative"
        else:
            return "neutral"

    df["sentiment"] = df["clean_text"].apply(get_sentiment)

    def preprocess_tokens(text):
        tokens = word_tokenize(str(text).lower())
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        return tokens

    df["tokens"] = df["clean_text"].apply(preprocess_tokens)

    # Pain point extraction (negative mentions)
    negative_texts = df[df["sentiment"] == "negative"]
    all_unigrams = [tok for tokens in negative_texts["tokens"] for tok in tokens]
    unigram_counts = Counter(all_unigrams).most_common(15)

    all_bigrams = [bg for tokens in negative_texts["tokens"] for bg in bigrams(tokens)]
    bigram_counts = Counter(all_bigrams).most_common(10)

    columns_to_keep = ["id", "category", "subreddit", "full_text", 
                       "keywords_found", "clean_text", "sentiment"]
    df_final = df[columns_to_keep].copy()
    df_final.to_csv(OUTPUT_FILE, index=False)

    log(f"Saved sentiment dataset with {len(df_final)} rows to {OUTPUT_FILE}.")
    log("Sentiment distribution:\n" + df_final["sentiment"].value_counts().to_string())
    log("Top negative keywords:\n" + pd.DataFrame(unigram_counts, columns=['word','count']).to_string(index=False))

    log("Sentiment analysis completed successfully.")

if __name__ == "__main__":
    main()
