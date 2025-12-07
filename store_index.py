"""
Index documents into Pinecone from data/reddit_data_sentiment.csv
Only indexes rows marked as unindexed in the tracker DB.
"""
import os
import sys
import ast
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Tracker DB helpers
from db import init_db, get_unindexed_ids, mark_indexed

load_dotenv()
ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"

print("=== Starting Pinecone Indexing ===")
init_db()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "reddit-insights")
DATA_PATH = DATA_DIR / "reddit_data_sentiment.csv"

if not DATA_PATH.exists():
    print(f"ERROR: Sentiment data not found: {DATA_PATH}")
    sys.exit(1)

if not PINECONE_API_KEY:
    print("ERROR: Missing PINECONE_API_KEY in environment.")
    sys.exit(1)

pc = Pinecone(api_key=PINECONE_API_KEY)

existing = [x["name"] for x in pc.list_indexes()]
if INDEX_NAME not in existing:
    print(f"Creating index {INDEX_NAME} (1536 dims)...")
    pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
else:
    print(f"Connected to Pinecone index: {INDEX_NAME}")

df = pd.read_csv(DATA_PATH, dtype=str).fillna("")
unindexed_ids = set(get_unindexed_ids() or [])

print(f"Rows in sentiment file: {len(df)}")
print(f"Unindexed IDs in tracker DB: {len(unindexed_ids)}")

# Filter rows to index
df_new = df[df["id"].isin(unindexed_ids)] if unindexed_ids else pd.DataFrame()
if df_new.empty:
    print("No new rows to index. Exiting.")
    sys.exit(0)

print(f"Preparing {len(df_new)} documents for indexing...")

docs = []
for _, row in df_new.iterrows():
    text = str(row.get("clean_text", "")).strip()
    if not text:
        continue
    try:
        kw = ast.literal_eval(row.get("keywords_found", "[]"))
        if not isinstance(kw, list):
            kw = []
    except Exception:
        kw = []

    meta = {
        "id": row.get("id", ""),
        "category": row.get("final_category", row.get("category", "")),
        "subreddit": row.get("subreddit", ""),
        "sentiment": row.get("sentiment", ""),
        "keywords": ", ".join(kw),
        "created_utc": row.get("created_utc", "")
    }
    docs.append(Document(page_content=text, metadata=meta))

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=650,
    chunk_overlap=120,
    separators=["\n\n", "\n", ".", "!", "?", " "]
)
chunks = splitter.split_documents(docs)
print(f"Generated {len(chunks)} chunks.")

# Embeddings
print("Initializing OpenAI embeddings (text-embedding-3-small)...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("Uploading vectors to Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("Upload complete. marking originals as indexed...")
count = 0
for d in docs:
    _id = d.metadata.get("id")
    if not _id:
        continue
    try:
        mark_indexed(_id)
        count += 1
    except Exception:
        pass

print(f"Indexed {count} original documents.")
print("=== Pinecone indexing finished ===")
