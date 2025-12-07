# render_app/app.py (updated to match local RAG behavior)
import os
from flask import Flask, request, render_template
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "reddit-insights")

app = Flask(__name__, static_folder="static", template_folder="templates")

# init
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
pc = None
docsearch = None
retriever = None
llm = None

# Retrieval configuration (same as local defaults)
EVIDENCE_SCORE_THRESH = float(os.getenv("EVIDENCE_SCORE_THRESH", 0.12))
MIN_DOCS_FOR_CONFIDENCE = int(os.getenv("MIN_DOCS_FOR_CONFIDENCE", 3))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 12))
SIM_SEARCH_K = int(os.getenv("SIM_SEARCH_K", 20))

try:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY missing")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [x["name"] for x in pc.list_indexes()]
    if INDEX_NAME in existing:
        docsearch = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
        retriever = docsearch.as_retriever(search_kwargs={"k": RETRIEVER_K})
        llm_model = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=llm_model, temperature=float(os.getenv("EVAL_LLM_TEMP", 0.2)))
        print("Render app: connected to pinecone & LLM.")
    else:
        print(f"Render app: index {INDEX_NAME} not found.")
except Exception as e:
    print("Render init failed:", e)
    pc = None
    docsearch = None
    retriever = None
    llm = None

# reuse helper functions similar to local (safe_truncate, format_doc_for_context, build_context, hybrid_retrieve, rag_generate, clean_model_output)
# (for brevity in this file, reimplement minimal versions — copy exact implementations from local app)
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
    if meta.get("keywords"): tags.append(f"tools: {meta.get('keywords')}")
    if meta.get("category"): tags.append(f"category: {meta.get('category')}")
    if meta.get("sentiment"): tags.append(f"sentiment: {meta.get('sentiment')}")
    if meta.get("subreddit"): tags.append(f"subreddit: {meta.get('subreddit')}")
    tag_str = " | ".join(tags) if tags else "metadata: none"
    return f"[{tag_str}]\n{excerpt}"

def build_context(docs, max_chars=8000):
    parts = [format_doc_for_context(d) for d in docs]
    context = "\n\n---\n\n".join(parts)
    if len(context) > max_chars:
        truncated = []
        curr = 0
        for p in parts:
            if curr + len(p) + 6 > max_chars:
                break
            truncated.append(p)
            curr += len(p) + 6
        context = "\n\n---\n\n".join(truncated)
    return context

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
        seen = set(); final=[]
        for d in (sem_docs or []) + (maybe_kw or []):
            key = (getattr(d, "page_content","") or "")[:400]
            if key and key not in seen:
                seen.add(key); final.append((d, None))
        return final
    else:
        seen = set(); final=[]
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        for d,s in doc_score_pairs:
            key = (getattr(d, "page_content","") or "")[:400]
            if key and key not in seen:
                seen.add(key); final.append((d,s))
        for d in (sem_docs or []):
            key = (getattr(d, "page_content","") or "")[:400]
            if key and key not in seen:
                seen.add(key); final.append((d,None))
        return final

import re, json
def clean_model_output(raw: str) -> str:
    if not raw: return ""
    s = raw.strip()
    try:
        js = json.loads(s)
        if isinstance(js, dict) and "answer" in js:
            s = js["answer"].strip()
    except Exception:
        pass
    s = s.replace("\r\n","\n").replace("\r","\n")
    s = re.sub(r"(?m)^###\s*Summary\s*$", "<h3>Summary</h3>", s)
    s = re.sub(r"(?m)^###\s*Key Points\s*$", "<h3>Key Points</h3>", s)
    s = re.sub(r"(?m)^###\s*Evidence Source.*$", "<h3>Evidence Source</h3>", s)
    lines = s.split("\n")
    out_lines=[]; in_list=False
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("- "):
            if not in_list:
                out_lines.append("<ul>"); in_list=True
            item = stripped[2:].strip()
            item = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", item)
            out_lines.append(f"<li>{item}</li>")
        else:
            if in_list:
                out_lines.append("</ul>"); in_list=False
            ln_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", ln)
            if ln_html.strip()=="":
                out_lines.append("<br/>")
            else:
                out_lines.append(f"<p>{ln_html}</p>")
    if in_list: out_lines.append("</ul>")
    out = "\n".join(out_lines)
    out = re.sub(r"(<br\/>\s*){2,}", "<br/>", out)
    return out.strip()

def rag_generate_render(query, doc_score_pairs):
    if not doc_score_pairs:
        return "I don't know based on the provided Reddit data."
    numeric_scores = [s for (_, s) in doc_score_pairs if s is not None]
    avg_score = float(sum(numeric_scores)/len(numeric_scores)) if numeric_scores else None
    num_docs = len(doc_score_pairs)
    if num_docs < 1: return "I don't know based on the provided Reddit data."
    if numeric_scores and avg_score < EVIDENCE_SCORE_THRESH and num_docs < MIN_DOCS_FOR_CONFIDENCE:
        return "I don't know based on the provided Reddit data."
    docs_only = [d for (d,_) in doc_score_pairs]
    context = build_context(docs_only)
    subreddits = sorted({(d.metadata or {}).get("subreddit","") for d in docs_only if (d.metadata or {}).get("subreddit")})
    sub_list = ", ".join([s for s in subreddits if s]) or "unknown"
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
    messages = [{"role":"system","content":system_prompt},{"role":"user","content":query}]
    try:
        resp = llm.invoke(messages)
        return resp.content.strip()
    except Exception:
        try:
            simple = " ".join([safe_truncate(getattr(d,"page_content","") or "", 400) for (d,_) in doc_score_pairs[:4]])
            fallback_prompt = [{"role":"system","content":"You are a concise assistant. Use ONLY the provided context."},
                               {"role":"user","content":f"Context:\n{simple}\n\nQuestion: {query}"}]
            r = llm.invoke(fallback_prompt)
            return r.content.strip()
        except Exception:
            return "I don't know based on the provided Reddit data."

@app.route("/")
def chat_page():
    return render_template("chatbot.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg","").strip()
    if not msg:
        return "Please enter a message."
    if docsearch is None or retriever is None or llm is None:
        return "Pinecone index not ready. Please run the local pipeline first."
    # greet
    if msg.lower() in {"hi","hello","hey","hiya"}:
        return "Hello! How can I help you today?"
    try:
        doc_score_pairs = hybrid_retrieve(msg)
    except Exception as e:
        return f"Retrieval error: {e}"
    if not doc_score_pairs:
        return "I don't know based on the provided Reddit data."
    try:
        ans = rag_generate_render(msg, doc_score_pairs)
        cleaned = clean_model_output(ans)
        return cleaned
    except Exception as e:
        return f"Error generating answer: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",8080)))
