import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

from dotenv import load_dotenv

# Embedding backends
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Optional OpenAI embeddings (langchain wrapper)
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None

load_dotenv()

ROOT = Path(".").resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = DATA_DIR / "reddit_data.csv"
BASELINE_OUT = DATA_DIR / "reddit_data_clean.csv"
SEMANTIC_OUT = DATA_DIR / "reddit_data_semantic_clean.csv"
AUDIT_OUT = DATA_DIR / "clean_data_audit.json"
LOG_FILE = DATA_DIR / "data_clean_log.txt"

KEYWORDS_PATH = Path(os.getenv("KEYWORDS_FILE", "config/keywords.json"))
SEMANTIC_MODEL_NAME = os.getenv("SEMANTIC_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "hf").lower()  # "hf" or "openai"
SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIM_THRESHOLD", 0.35))
MIN_WORDS = int(os.getenv("MIN_WORDS_CLEAN", 5))
KEYWORD_MATCH_REQUIRED = os.getenv("KEYWORD_MATCH_REQUIRED", "false").lower() == "true"

#utils
def log(msg: str):
    ts = datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # fallback to stdout only
        pass

def clean_text_to_alpha(text: Optional[str]) -> str:
    if text is None:
        return ""
    t = str(text)
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[^A-Za-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()

def simple_token_count(text: str) -> int:
    if not isinstance(text, str) or text.strip() == "":
        return 0
    return len(text.split())

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def load_keywords(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        log(f"keywords.json not found at {path}; using empty sets.")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # normalize lower-case
        return {k: [w.lower() for w in v] for k, v in data.items()}
    except Exception as e:
        log(f"Failed to load keywords.json: {e}")
        return {}

#main
def main():
    log("Start combined cleaning & semantic classification")

    if not RAW_FILE.exists():
        log(f"ERROR: Missing input file: {RAW_FILE}")
        sys.exit(1)

    # Load raw CSV
    df = pd.read_csv(RAW_FILE, dtype=str).fillna("")
    log(f"Loaded raw rows: {len(df)}")

    # Deduplicate by id if present
    if "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["id"], keep="last")
        log(f"Dropped duplicates by id: {before - len(df)}")

    # Compose full_text if missing
    if "full_text" not in df.columns:
        left = df.columns.intersection(["title", "content"])
        if len(left) >= 1:
            df["full_text"] = df[["title", "content"]].astype(str).agg(" ".join, axis=1)
        else:
            df["full_text"] = df.astype(str).agg(" ".join, axis=1)

    # Load keywords
    keywords = load_keywords(KEYWORDS_PATH)
    if not keywords:
        # fallback minimal
        keywords = {
            "Law": ["clio", "lexisnexis", "westlaw"],
            "Construction": ["procore", "autocad", "revit"],
            "Tech": ["docker", "kubernetes", "aws"]
        }

    # baseline keyword matching
    def find_keywords_in_text(text: str):
        t = text.lower()
        found = []
        for cat, kws in keywords.items():
            for kw in kws:
                if re.search(rf"\b{re.escape(kw)}\b", t):
                    found.append(kw)
        return sorted(set(found))

    df["keywords_found"] = df["full_text"].apply(find_keywords_in_text)

    # baseline cleaning
    df["clean_text"] = df["full_text"].apply(clean_text_to_alpha)
    df["word_count"] = df["clean_text"].apply(simple_token_count)
    df["drop_reason"] = df["word_count"].apply(lambda wc: "too_short" if wc < MIN_WORDS else None)
    df.loc[df["clean_text"].str.strip() == "", "drop_reason"] = "noise"

    baseline_df = df[df["drop_reason"].isna()].copy()
    if KEYWORD_MATCH_REQUIRED:
        baseline_df = baseline_df[baseline_df["keywords_found"].map(len) > 0].copy()

    # save baseline
    baseline_df.to_csv(BASELINE_OUT, index=False, encoding="utf-8")
    log(f"Baseline cleaned: {len(baseline_df)} rows -> {BASELINE_OUT}")

    # If no baseline rows, finish
    if baseline_df.empty:
        produce_audit(df, baseline_df, pd.DataFrame())
        log("No baseline rows to process for semantic classification. Exiting.")
        return

    # Prepare category prototypes from keyword lists (join keywords)
    CATEGORY_PROTOTYPES = {cat: " ".join(kws) for cat, kws in keywords.items()}

    # Build embeddings depending on backend
    use_openai = EMBEDDING_BACKEND == "openai"
    if use_openai:
        if OpenAIEmbeddings is None:
            log("OpenAIEmbeddings not available in environment. Please install langchain_openai or set EMBEDDING_BACKEND=hf")
            produce_audit(df, baseline_df, pd.DataFrame())
            return
        log("Using OpenAI embeddings for semantic classification.")
        emb_client = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
        # OpenAIEmbeddings provides embed_documents / embed_query; we'll use embed_documents
        def embed_texts(texts: List[str]) -> List[List[float]]:
            return emb_client.embed_documents(texts)
        def embed_query(text: str) -> List[float]:
            return emb_client.embed_query(text)
    else:
        if SentenceTransformer is None:
            log("sentence-transformers not installed. Install it or set EMBEDDING_BACKEND=openai")
            produce_audit(df, baseline_df, pd.DataFrame())
            return
        log(f"Loading HuggingFace model: {SEMANTIC_MODEL_NAME}")
        model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        def embed_texts(texts: List[str]) -> List[List[float]]:
            # chunk to avoid memory pressure
            out = []
            B = 2048
            for i in range(0, len(texts), B):
                batch = texts[i:i+B]
                out.extend(model.encode(batch, show_progress_bar=False, convert_to_numpy=True))
            return [list(map(float, v)) for v in out]
        def embed_query(text: str) -> List[float]:
            return list(map(float, model.encode([text])[0]))

    # build prototype embeddings
    proto_texts = [CATEGORY_PROTOTYPES[c] for c in CATEGORY_PROTOTYPES]
    try:
        proto_embs = embed_texts(proto_texts)
    except Exception as e:
        log(f"Failed to compute prototype embeddings: {e}")
        proto_embs = [embed_query(t) for t in proto_texts]
    proto_map = {cat: np.array(vec, dtype=float) for cat, vec in zip(CATEGORY_PROTOTYPES.keys(), proto_embs)}
    log(f"Prepared prototype embeddings for categories: {list(proto_map.keys())}")

    # compute embeddings for baseline rows (clean_text)
    sample_df = baseline_df.copy().reset_index(drop=True)
    texts = sample_df["clean_text"].astype(str).tolist()
    log(f"Computing embeddings for {len(texts)} baseline rows...")
    try:
        vectors = embed_texts(texts)
    except Exception as e:
        log(f"Embedding batch failed: {e}; falling back to per-item")
        vectors = [embed_query(t) for t in texts]
    vectors = [np.array(v, dtype=float) for v in vectors]
    log(f"Computed embeddings for {len(vectors)} rows.")

    # assign semantic category by cosine similarity to prototypes
    assigned = []
    scores = []
    for vec in vectors:
        best_cat = None
        best_score = -999.0
        for cat, pvec in proto_map.items():
            s = cosine_sim(vec, pvec)
            if s > best_score:
                best_score = s
                best_cat = cat
        scores.append(float(best_score))
        assigned.append(best_cat if best_score >= SIMILARITY_THRESHOLD else None)

    sample_df["semantic_category"] = assigned
    sample_df["semantic_score"] = scores

    # resolve final category: prefer explicit keyword mapping when present
    def resolve_final_category(row):
        kws = row.get("keywords_found") or []
        if kws:
            for kw in kws:
                kw_l = kw.lower()
                for cat, kwlist in keywords.items():
                    if kw_l in [k.lower() for k in kwlist]:
                        return cat
        return row.get("semantic_category")

    sample_df["final_category"] = sample_df.apply(resolve_final_category, axis=1)

    # keep rows that have final_category
    semantic_keep = sample_df[sample_df["final_category"].notna()].copy()
    semantic_keep.to_csv(SEMANTIC_OUT, index=False, encoding="utf-8")
    log(f"Semantic cleaned saved: {len(semantic_keep)} rows -> {SEMANTIC_OUT}")

    # produce audit
    produce_audit(df, baseline_df, semantic_keep)
    log("Finished successfully.")

def produce_audit(df_all: pd.DataFrame, baseline_df: pd.DataFrame, semantic_df: pd.DataFrame):
    total_raw = int(len(df_all))
    baseline_rows = int(len(baseline_df))
    semantic_rows = int(len(semantic_df))
    dropped = total_raw - baseline_rows
    reasons = df_all["drop_reason"].value_counts(dropna=True).to_dict()
    audit = {
        "timestamp": datetime.utcnow().isoformat(),
        "raw_rows": total_raw,
        "clean_rows": baseline_rows,
        "semantic_rows": semantic_rows,
        "dropped_rows": dropped,
        "reason_breakdown": {k: int(v) for k, v in reasons.items()}
    }
    with open(AUDIT_OUT, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    log(f"Audit saved to {AUDIT_OUT}")

if __name__ == "__main__":
    main()
