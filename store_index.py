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

# ----------------------------------------------------------------------
# Rebuild Pinecone index using OpenAI text-embedding-3-small (1536 dims)
# ----------------------------------------------------------------------

sys.stdout.reconfigure(line_buffering=True)
print("Initializing indexing...")

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    print("Missing Pinecone API key.")
    sys.exit(1)

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "reddit-insights"

DATA_PATH = "data/reddit_data_sentiment.csv"
if not os.path.exists(DATA_PATH):
    print(f"Missing sentiment file: {DATA_PATH}")
    sys.exit(1)

# Init DB tracker
init_db()
unindexed_ids = set(get_unindexed_ids())

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows; {len(unindexed_ids)} unindexed IDs.")

df_new = df[df["id"].isin(unindexed_ids)]
if df_new.empty:
    print("No new documents to index.")
    sys.exit(0)

# Prepare documents
docs = []
for _, row in df_new.iterrows():
    text = str(row.get("clean_text", "")).strip()
    if not text:
        continue

    keywords = []
    if isinstance(row.get("keywords_found"), str):
        try:
            keywords = ast.literal_eval(row["keywords_found"])
        except:
            keywords = [row["keywords_found"]]

    metadata = {
        "id": row["id"],
        "subreddit": row["subreddit"],
        "category": row["category"],
        "sentiment": row.get("sentiment", ""),
        "keywords": ", ".join(keywords),
    }

    docs.append(Document(page_content=text, metadata=metadata))

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)
chunks = splitter.split_documents(docs)

# ----------------------------------------------------------------------
# Using OpenAI text-embedding-3-small (1536 dimensional)
# ----------------------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
dimension = 1536

# ----------------------------------------------------------------------
# Create Pinecone index (new one)
# ----------------------------------------------------------------------
existing = [i["name"] for i in pc.list_indexes()]
if index_name not in existing:
    print("Creating new Pinecone index with 1536-dim embeddings...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print("Index already exists.")

print("Uploading embeddings...")
vectorstore = PineconeVectorStore.from_documents(
    chunks,
    index_name=index_name,
    embedding=embeddings
)

# Mark documents indexed
for doc in docs:
    mark_indexed(doc.metadata["id"])

print(f"Indexed {len(docs)} new documents successfully.")
