import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response

# LangChain + Pinecone clients
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Flask + env setup
app = Flask(__name__, static_folder="static", template_folder="templates")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "reddit-insights")

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Paths
ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
SCRAPE_LOG = DATA_DIR / "scrape_log.txt"

RAW_PATH = DATA_DIR / "reddit_data.csv"
CLEAN_PATH = DATA_DIR / "reddit_data_clean.csv"
SEMANTIC_PATH = DATA_DIR / "reddit_data_semantic_clean.csv"
SENT_PATH = DATA_DIR / "reddit_data_sentiment.csv"
EVAL_PATH = DATA_DIR / "evaluation_results.csv"

CONFIG_PATH = ROOT / "config" / "subreddits.json"

PIPELINE_STEPS = {
    "collect": "data_collection.py",
    "clean_semantic": "data_clean_and_classify.py",
    "sentiment": "data_sentiment.py",
    "index": "store_index.py",
    "evaluate": "evaluate.py"
}


# RAG configuration
EVIDENCE_SCORE_THRESH = float(os.getenv("EVIDENCE_SCORE_THRESH", 0.12))
MIN_DOCS_FOR_CONFIDENCE = int(os.getenv("MIN_DOCS_FOR_CONFIDENCE", 3))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 8000))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 12))
SIM_SEARCH_K = int(os.getenv("SIM_SEARCH_K", 20))


# Init Pinecone + VectorStore + LLM
docsearch = None
retriever = None
pc = None
llm = None

try:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not found in env")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    emb = OpenAIEmbeddings(model=os.getenv("EVAL_EMB_MODEL", "text-embedding-3-small"))

    docsearch = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=emb)
    retriever = docsearch.as_retriever(search_kwargs={"k": RETRIEVER_K})

    llm_model = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=llm_model, temperature=float(os.getenv("EVAL_LLM_TEMP", 0.2)))

    print("Connected to Pinecone index and initialized LLM.")
except Exception as e:
    print("Warning: failed to initialize Pinecone/LLM:", e)
    docsearch = None
    retriever = None
    pc = None
    llm = None

# Utility helpers
def safe_read_csv(path: Path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

def now_ts():
    return datetime.utcnow().isoformat()

def safe_truncate(text: str, chars: int):
    if not text:
        return ""
    text = str(text)
    if len(text) <= chars:
        return text
    cut = text[:chars]
    last_period = cut.rfind(".")
    if last_period > int(chars * 0.6):
        return cut[: last_period + 1]
    return cut

def format_doc_for_context(d):
    meta = d.metadata or {}
    excerpt = getattr(d, "page_content", "") or ""
    excerpt = " ".join(excerpt.split())
    excerpt = safe_truncate(excerpt, 800)
    tags = []
    if meta.get("keywords"):
        tags.append(f"tools: {meta.get('keywords')}")
    if meta.get("category"):
        tags.append(f"category: {meta.get('category')}")
    if meta.get("sentiment"):
        tags.append(f"sentiment: {meta.get('sentiment')}")
    if meta.get("subreddit"):
        tags.append(f"subreddit: {meta.get('subreddit')}")
    tag_str = " | ".join(tags) if tags else "metadata: none"
    return f"[{tag_str}]\n{excerpt}"

def build_context(docs):
    parts = [format_doc_for_context(d) for d in docs]
    context = "\n\n---\n\n".join(parts)
    if len(context) > MAX_CONTEXT_CHARS:
        truncated = []
        curr = 0
        for p in parts:
            if curr + len(p) + 6 > MAX_CONTEXT_CHARS:
                break
            truncated.append(p)
            curr += len(p) + 6
        context = "\n\n---\n\n".join(truncated)
    return context


# Hybrid retrieval
# returns list of (Document, score_or_none)

def hybrid_retrieve(query: str):
    if docsearch is None or retriever is None:
        return []

    doc_score_pairs = []
    try:
        sim_with_score = getattr(docsearch, "similarity_search_with_score", None)
        if callable(sim_with_score):
            pairs = docsearch.similarity_search_with_score(query, k=SIM_SEARCH_K)
            if pairs:
                for item in pairs:
                    if isinstance(item, tuple) and len(item) >= 2:
                        d, s = item[0], float(item[1])
                        doc_score_pairs.append((d, s))
    except Exception:
        doc_score_pairs = []

    try:
        sem_docs = retriever.get_relevant_documents(query)
    except Exception:
        sem_docs = []

    if not doc_score_pairs:
        try:
            maybe_kw = docsearch.similarity_search(query, k=SIM_SEARCH_K)
        except Exception:
            maybe_kw = []
        seen = set()
        final = []
        for d in (sem_docs or []) + (maybe_kw or []):
            content = getattr(d, "page_content", "") or ""
            key = content[:400]
            if key and key not in seen:
                seen.add(key)
                final.append((d, None))
        return final
    else:
        seen = set()
        final = []
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        for d, s in doc_score_pairs:
            key = (getattr(d, "page_content", "") or "")[:400]
            if key and key not in seen:
                seen.add(key)
                final.append((d, s))
        for d in (sem_docs or []):
            key = (getattr(d, "page_content", "") or "")[:400]
            if key and key not in seen:
                seen.add(key)
                final.append((d, None))
        return final


# RAG generation
def rag_generate(query: str, doc_score_pairs):
    if not doc_score_pairs:
        return "I don't know based on the provided Reddit data."

    numeric_scores = [s for (_, s) in doc_score_pairs if s is not None]
    avg_score = float(sum(numeric_scores) / len(numeric_scores)) if numeric_scores else None
    num_docs = len(doc_score_pairs)

    if num_docs < 1:
        return "I don't know based on the provided Reddit data."
    if numeric_scores:
        if avg_score < EVIDENCE_SCORE_THRESH and num_docs < MIN_DOCS_FOR_CONFIDENCE:
            return "I don't know based on the provided Reddit data."

    docs_only = [d for (d, _) in doc_score_pairs]
    context = build_context(docs_only)
    subreddits = sorted({(d.metadata or {}).get("subreddit", "") for d in docs_only if (d.metadata or {}).get("subreddit")})
    sub_list = ", ".join([s for s in subreddits if s]) or "unknown"

    # Exact system prompt (from evaluate.py) + greeting special-case note
    system_prompt = (
        "You are a helpful and precise research assistant. Your primary job is to answer "
        "questions *grounded strictly in the provided Reddit excerpts (the 'Context')*.\n"
        "However, if the user is simply greeting you (e.g., 'hi', 'hello', 'hey'), respond "
        "politely without requiring context — do NOT say you don't know.\n\n"

        "STRICT RULES:\n"
        "1) For informational questions: Use ONLY the provided context.\n"
        "2) If the context does *not* contain enough information to answer confidently, reply EXACTLY:\n"
        "   \"I don't know based on the provided Reddit data.\"\n"
        "3) When answering with context, ALWAYS return the following structure:\n\n"

        "### Summary\n"
        "- 2–6 line concise explanation.\n\n"

        "### Key Points\n"
        "- Bullet list of key tools, issues, or sentiments mentioned.\n"
        "- ALWAYS include sentiment for each item when available.\n\n"

        "### Evidence Source\n"
        "- One line stating which subreddits the evidence came from.\n\n"

        f"Context (excerpted from subreddits: {sub_list}):\n{context}\n\n"
        "Produce clear, structured, factual responses.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    try:
        resp = llm.invoke(messages)
        return resp.content.strip()
    except Exception:
        try:
            simple = " ".join([safe_truncate(getattr(d, "page_content", "") or "", 400) for (d, _) in doc_score_pairs[:4]])
            fallback_prompt = [
                {"role": "system", "content": "You are a concise assistant. Use ONLY the provided context."},
                {"role": "user", "content": f"Context:\n{simple}\n\nQuestion: {query}"}
            ]
            r = llm.invoke(fallback_prompt)
            return r.content.strip()
        except Exception:
            return "I don't know based on the provided Reddit data."

#strip JSON wrapper
import re
def clean_model_output(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()

    # If output looks like JSON wrapper try extract
    try:
        # allow bare JSON or newline-prefixed JSON
        js = json.loads(s)
        if isinstance(js, dict) and "answer" in js:
            s = js["answer"].strip()
    except Exception:
        # not JSON — proceed
        pass

    # Convert simple markdown headings and lists to HTML
    # Highest priority: Sections starting with "### Summary", "### Key Points", "### Evidence Source"
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Replace headings
    s = re.sub(r"(?m)^###\s*Summary\s*$", "<h3>Summary</h3>", s)
    s = re.sub(r"(?m)^###\s*Key Points\s*$", "<h3>Key Points</h3>", s)
    s = re.sub(r"(?m)^###\s*Evidence Source\s*$", "<h3>Evidence Source</h3>", s)
    s = re.sub(r"(?m)^###\s*Evidence Source.*$", "<h3>Evidence Source</h3>", s)

    # Convert bullet lists starting with lines beginning with "-" to <ul><li>...
    lines = s.split("\n")
    out_lines = []
    in_list = False
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("- "):
            if not in_list:
                out_lines.append("<ul>")
                in_list = True
            item = stripped[2:].strip()
            # Allow bold markers **...**
            item = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", item)
            out_lines.append(f"<li>{item}</li>")
        else:
            if in_list:
                out_lines.append("</ul>")
                in_list = False
            # simple bold handling for inline "**text**"
            ln_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", ln)
            # preserve blank lines as paragraph breaks
            if ln_html.strip() == "":
                out_lines.append("<br/>")
            else:
                out_lines.append(f"<p>{ln_html}</p>")
    if in_list:
        out_lines.append("</ul>")
    out = "\n".join(out_lines)

    # basic cleanup: collapse multiple <br/> sequences
    out = re.sub(r"(<br\/>\s*){2,}", "<br/>", out)

    # Trim outer whitespace
    return out.strip()


# Flask routes — pages
@app.route("/")
def dashboard_page():
    return render_template("dashboard.html")

@app.route("/subreddits")
def subs_page():
    return render_template("manager.html")

@app.route("/pipeline")
def pipeline_page():
    return render_template("pipeline.html")

@app.route("/chat")
def chat_page():
    return render_template("chatbot.html")

# API — subreddits
@app.route("/api/subreddits", methods=["GET"])
def api_get_subs():
    try:
        if not CONFIG_PATH.exists():
            default = {"Law": [], "Construction": [], "Tech": []}
            CONFIG_PATH.write_text(json.dumps(default, indent=2))
            return jsonify(default)
        return jsonify(json.loads(CONFIG_PATH.read_text()))
    except Exception:
        return jsonify({"Law": [], "Construction": [], "Tech": []})

@app.route("/api/subreddits", methods=["POST"])
def api_save_subs():
    data = request.get_json() or {}
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, indent=2))
    return jsonify({"status": "ok"})


# API — dashboard
@app.route("/api/dashboard")
def api_dashboard():
    def stats(path: Path):
        return {
            "exists": path.exists(),
            "rows": int(len(safe_read_csv(path))) if path.exists() else 0,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None
        }

    tail = []
    if SCRAPE_LOG.exists():
        try:
            with open(SCRAPE_LOG, "r", encoding="utf-8") as f:
                tail = f.readlines()[-50:]
        except Exception:
            tail = []

    return jsonify({
        "raw": stats(RAW_PATH),
        "clean": stats(CLEAN_PATH),
        "semantic": stats(SEMANTIC_PATH),
        "sentiment": stats(SENT_PATH),
        "evaluation": stats(EVAL_PATH),
        "last_log_tail": tail
    })

# SSE: run pipeline scripts and stream output
def stream_process(script_path: str):
    script = ROOT / script_path
    if not script.exists():
        yield f"data: ERROR — script not found: {script_path}\n\n"
        yield "event: close\n\n"
        return

    p = None
    try:
        p = __import__("subprocess").Popen(
            [sys.executable, "-u", str(script)],
            stdout=__import__("subprocess").PIPE,
            stderr=__import__("subprocess").STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        for line in iter(p.stdout.readline, ""):
            yield f"data: {line.rstrip()}\n\n"
    except Exception as exc:
        yield f"data: ERROR running script: {exc}\n\n"
    finally:
        try:
            if p:
                p.stdout.close()
                p.wait()
        except Exception:
            pass
        yield f"data: Step finished ({script_path})\n\n"
        yield "event: close\n\n"

@app.route("/stream/<step>")
def run_single_step(step):
    if step not in PIPELINE_STEPS:
        return "Invalid step", 400
    return Response(stream_process(PIPELINE_STEPS[step]), mimetype="text/event-stream")

@app.route("/stream/full")
def run_full_pipeline():
    def gen():
        for name, script in PIPELINE_STEPS.items():
            yield f"data: === Starting {name} ===\n\n"
            for line in stream_process(script):
                yield line
            yield f"data: === Finished {name} ===\n\n"
            time.sleep(0.25)
        yield "event: close\n\n"
    return Response(gen(), mimetype="text/event-stream")


# CHAT endpoint
# returns cleaned HTML
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "No message provided.", 400

    if docsearch is None or llm is None:
        return "Index or LLM not available — run indexing or check env.", 503

    # quick greet handling — allow short polite reply without retrieval
    if msg.lower() in {"hi", "hello", "hey", "hiya", "good morning", "good afternoon", "good evening"}:
        return "Hello! How can I help you today?"

    try:
        doc_score_pairs = hybrid_retrieve(msg)
    except Exception as e:
        return f"Retrieval error: {e}", 500

    if not doc_score_pairs:
        return "I don't know based on the provided Reddit data."

    try:
        answer = rag_generate(msg, doc_score_pairs)
        cleaned = clean_model_output(answer)
    except Exception as e:
        return f"Generation error: {e}", 500

    # Return HTML
    return cleaned

# Run
if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)
