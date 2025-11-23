import os
import sys
import time
import json
import subprocess
import pandas as pd
from datetime import datetime

from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv

# --- LangChain + Pinecone ---
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec


# ----------------------------------------
# FLASK
# ----------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATA_DIR = "data"
SCRAPE_LOG = os.path.join(DATA_DIR, "scrape_log.txt")

RAW_PATH = os.path.join(DATA_DIR, "reddit_data.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "reddit_data_clean.csv")
SENT_PATH = os.path.join(DATA_DIR, "reddit_data_sentiment.csv")
EVAL_PATH = os.path.join(DATA_DIR, "evaluation_results.csv")

CONFIG_PATH = "config/subreddits.json"


# ----------------------------------------
# RAG SETUP (Switched to OpenAI Embeddings)
# ----------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

index_name = "reddit-insights"

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [i["name"] for i in pc.list_indexes()]

    if index_name not in existing_indexes:
        print("Creating Pinecone index...")
        pc.create_index(
            name=index_name,
            dimension=1536,   # OpenAI embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(10)
    else:
        print("Connected to Pinecone index.")

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

except Exception as e:
    print("Could not connect to Pinecone:", e)
    docsearch = None

retriever = docsearch.as_retriever(search_kwargs={"k": 10}) if docsearch else None


# ----------------------------------------
# LLM + Prompt
# ----------------------------------------
system_prompt = (
    "You are a research assistant summarizing Reddit discussions about software tools "
    "used in Law, Construction, and Tech industries.\n\n"
    "Use Reddit excerpts to answer accurately. If insufficient context, reply:\n"
    "'I don’t know based on the provided Reddit data.'\n\n"
    "Context:\n{context}"
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=400)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain) if retriever else None


# ----------------------------------------
# PIPELINE SCRIPTS
# ----------------------------------------
PIPELINE_STEPS = {
    "collect": "data_collection.py",
    "clean": "data_clean.py",
    "sentiment": "data_sentiment.py",
    "index": "store_index.py",
    "evaluate": "evaluate.py"
}


# ----------------------------------------
# HELPERS
# ----------------------------------------
def safe_read_csv(path):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def load_subreddits():
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump({"Law": [], "Construction": [], "Tech": []}, f, indent=2)
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_subreddits(data):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ----------------------------------------
# STREAM PROCESS
# ----------------------------------------
def stream_process(script_path):
    process = subprocess.Popen(
        [sys.executable, "-u", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for line in iter(process.stdout.readline, ""):
        yield f"data:{line.strip()}\n\n"
    yield "event: close\ndata: done\n\n"


# ----------------------------------------
# ROUTES (UI)
# ----------------------------------------
@app.route("/")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/subreddits")
def manager_page():
    return render_template("manager.html")


@app.route("/pipeline")
def pipeline_page():
    return render_template("pipeline.html")


@app.route("/chat")
def chat_page():
    return render_template("chatbot.html")


# ----------------------------------------
# SUBREDDIT API
# ----------------------------------------
@app.route("/api/subreddits", methods=["GET"])
def api_get_subs():
    return jsonify(load_subreddits())


@app.route("/api/subreddits", methods=["POST"])
def api_save_subs():
    save_subreddits(request.get_json())
    return jsonify({"status": "ok"})


# ----------------------------------------
# CHATBOT ENDPOINT
# ----------------------------------------
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]

    if not docsearch:
        return "Index missing — run indexing first."

    # Simple category filter
    text = msg.lower()
    if "construction" in text:
        search_filter = {"category": "Construction"}
    elif "law" in text or "legal" in text:
        search_filter = {"category": "Law"}
    elif "tech" in text or "software" in text:
        search_filter = {"category": "Tech"}
    else:
        search_filter = None

    retrieved = docsearch.similarity_search_with_score(msg, k=10, filter=search_filter)
    docs = [doc for doc, score in retrieved if doc.page_content.strip()]

    if not docs:
        return "I don’t know based on the provided Reddit data."

    response = question_answer_chain.invoke({"input": msg, "context": docs})

    answer = response.strip() if isinstance(response, str) else str(response)

    if not answer:
        return "I don’t know based on the provided Reddit data."

    return answer


# ----------------------------------------
# RUN
# ----------------------------------------
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=True)
