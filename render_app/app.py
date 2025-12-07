# render_app/app.py

import os
import json
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from pathlib import Path

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

#ENV variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "reddit-insights")

#FLASK Loading 
app = Flask(__name__, static_folder="static", template_folder="templates")

#DEFAULTS Values
EVIDENCE_SCORE_THRESH = 0.12
MIN_DOCS_FOR_CONFIDENCE = 3
MAX_CONTEXT_CHARS = 8000
RETRIEVER_K = 12
SIM_SEARCH_K = 20

#INIT
pc = None
docsearch = None
retriever = None
llm = None

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    emb_model = "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(model=emb_model)

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_kwargs={"k": RETRIEVER_K})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    print("[Render] Pinecone + LLM initialized.")
except Exception as e:
    print("[Render] Initialization error:", e)
    docsearch = None
    retriever = None
    llm = None


#UTILITIES
def safe_truncate(text, max_chars):
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def format_doc(d):
    meta = d.metadata or {}
    excerpt = safe_truncate((d.page_content or "").replace("\n", " "), 600)

    parts = []
    if meta.get("keywords"):  parts.append(f"tools: {meta['keywords']}")
    if meta.get("category"):  parts.append(f"category: {meta['category']}")
    if meta.get("sentiment"): parts.append(f"sentiment: {meta['sentiment']}")
    if meta.get("subreddit"): parts.append(f"subreddit: {meta['subreddit']}")

    tag = " | ".join(parts) if parts else "metadata: none"
    return f"[{tag}]\n{excerpt}"

def build_context(docs):
    blocks = [format_doc(d) for d in docs]
    ctx = "\n\n---\n\n".join(blocks)

    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[:MAX_CONTEXT_CHARS]
    return ctx


#HYBRID RETRIEVAL
def hybrid_retrieve(query):
    if not docsearch:
        return []

    results = []

    # Try scored similarity
    try:
        scored = docsearch.similarity_search_with_score(query, k=SIM_SEARCH_K)
        for doc, score in scored:
            results.append((doc, float(score)))
    except:
        scored = []

    # Add semantic retriever docs
    try:
        sem_docs = retriever.get_relevant_documents(query)
    except:
        sem_docs = []

    seen = set()
    final = []

    # scored first
    for d, s in results:
        key = d.page_content[:300]
        if key not in seen:
            seen.add(key)
            final.append((d, s))

    # then sem docs
    for d in sem_docs:
        key = d.page_content[:300]
        if key not in seen:
            seen.add(key)
            final.append((d, None))

    return final


#RAG GENERATION
def rag_generate(query, doc_score_pairs):

    # greeting case
    if query.lower() in ["hi","hello","hey","yo","sup","hi there"]:
        return "Hello! 👋 How can I help you today?"

    if not doc_score_pairs:
        return "I don't know based on the provided Reddit data."

    numeric_scores = [s for (_, s) in doc_score_pairs if s is not None]
    avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else None

    # check confidence
    if numeric_scores and avg_score < EVIDENCE_SCORE_THRESH and len(doc_score_pairs) < MIN_DOCS_FOR_CONFIDENCE:
        return "I don't know based on the provided Reddit data."

    docs = [d for d, _ in doc_score_pairs]
    context = build_context(docs)

    subreddits = sorted({(d.metadata or {}).get("subreddit","") for d in docs})
    sub_list = ", ".join([s for s in subreddits if s]) or "unknown"

    system_prompt = f"""
You are a helpful and precise research assistant. Your primary job is to answer 
questions grounded strictly in the provided Reddit excerpts (the 'Context')
However, if the user is simply greeting you (e.g., 'hi', 'hello', 'hey'), respond
politely without requiring context — do NOT say you don't know.
STRICT RULES:
1) For informational questions: Use ONLY the provided context.
2) If the context does *not* contain enough information to answer confidently, reply EXACTLY:
    I don't know based on the provided Reddit data.\
3) When answering with context, ALWAYS return the following structure:


### Summary
(2–6 line concise explanation)

### Key Points
- Bullet list of key tools, issues, or sentiments mentioned.
- ALWAYS include sentiment for each item when available.


### Evidence Source
- Subreddits: {sub_list}
Produce clear, structured, factual responses.
Context:
{context}
"""

    try:
        resp = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ])
        return resp.content.strip()
    except:
        return "I don't know based on the provided Reddit data."


# --------------------- ROUTES ---------------------------
@app.route("/")
def home():
    return render_template("chatbot.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg","").strip()

    if not msg:
        return jsonify({"answer": "Please enter a message."})

    if not docsearch or not llm:
        return jsonify({"answer": "Index or LLM not available."})

    try:
        docs = hybrid_retrieve(msg)
        answer = rag_generate(msg, docs)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Server error: {str(e)}"})


# --------------------- MAIN -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

