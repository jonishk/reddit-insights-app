import os
import sys
import ast
import pandas as pd
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from db import init_db, get_unindexed_ids, mark_indexed

# -----------------------------
# Setup
# -----------------------------
sys.stdout.reconfigure(line_buffering=True)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    print("Missing Pinecone API key.")
    sys.exit(1)

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "reddit-insights"

DATA_PATH = "data/reddit_data_sentiment.csv"
if not os.path.exists(DATA_PATH):
    print(f"File not found: {DATA_PATH}")
    sys.exit(1)

init_db()

df = pd.read_csv(DATA_PATH)
unindexed_ids = set(get_unindexed_ids())

print(f"Loaded {len(df)} total rows; {len(unindexed_ids)} unindexed IDs.")

# -----------------------------
# Filter new rows
# -----------------------------
df_new = df[df["id"].isin(unindexed_ids)]
if df_new.empty:
    print("No new data to index.")
    sys.exit(0)

print(f"Preparing {len(df_new)} documents...")

docs = []
for _, row in df_new.iterrows():
    text = str(row.get("clean_text", "")).strip()
    if not text or text.lower() == "nan":
        continue

    keywords = []
    if isinstance(row.get("keywords_found"), str):
        try:
            keywords = ast.literal_eval(row["keywords_found"])
        except Exception:
            keywords = [row["keywords_found"]]

    metadata = {
        "id": row.get("id", ""),
        "category": row.get("category", ""),
        "subreddit": row.get("subreddit", ""),
        "keywords": ", ".join(keywords) if keywords else "",
        "sentiment": row.get("sentiment", "")
    }

    docs.append(Document(page_content=text, metadata=metadata))

# -----------------------------
# Chunking
# -----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)
doc_chunks = splitter.split_documents(docs)

# -----------------------------
# OpenAI embeddings (new)
# -----------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Pinecone index — use 1536 dims for text-embedding-3-small
if index_name not in [i["name"] for i in pc.list_indexes()]:
    print("Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Index created.")

vectorstore = PineconeVectorStore.from_documents(
    documents=doc_chunks,
    index_name=index_name,
    embedding=embeddings,
)

for doc in docs:
    mark_indexed(doc.metadata["id"])

print(f"Indexed {len(docs)} documents successfully.")
