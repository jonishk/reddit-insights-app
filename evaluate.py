import os
import json
import math
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Config
load_dotenv()

DATA_DIR = Path("data")
CONFIG_DIR = Path("config")
QUESTIONS_FILE = CONFIG_DIR / "questions.json"
OUTPUT_FILE = DATA_DIR / "evaluation_results.csv"

INDEX_NAME = os.getenv("PINECONE_INDEX", "reddit-insights")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
EMB_MODEL = os.getenv("EVAL_EMB_MODEL", "text-embedding-3-small")

# Evidence thresholds
EVIDENCE_SCORE_THRESH = float(os.getenv("EVIDENCE_SCORE_THRESH", 0.12))
MIN_DOCS_FOR_CONFIDENCE = int(os.getenv("MIN_DOCS_FOR_CONFIDENCE", 3))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 8000))

# How many candidates to retrieve
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 12))
SIM_SEARCH_K = int(os.getenv("SIM_SEARCH_K", 20))


# Init clients
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing in env")

pc = Pinecone(api_key=PINECONE_API_KEY)
emb = OpenAIEmbeddings(model=EMB_MODEL)

# Pinecone-based vectorstore
vs = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=emb)

retriever = vs.as_retriever(search_kwargs={"k": RETRIEVER_K})
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

# Utilities
def now_ts():
    return datetime.utcnow().isoformat()

def load_questions():
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing questions.json at {QUESTIONS_FILE}")
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

def safe_truncate(text: str, chars: int):
    if not text:
        return ""
    if len(text) <= chars:
        return text
    # try to preserve whole sentences if possible
    cut = text[:chars]
    last_period = cut.rfind(".")
    if last_period > int(chars * 0.6):
        return cut[: last_period + 1]
    return cut

def format_doc_for_context(d):
    """
    Build a compact representation of a Document for context:
    [tags] excerpt...
    """
    meta = d.metadata or {}
    excerpt = getattr(d, "page_content", "") or ""
    excerpt = " ".join(excerpt.split())  # collapse whitespace
    excerpt = safe_truncate(excerpt, 800)  # per-document excerpt limit
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
    """
    Join formatted doc excerpts into one context string and ensure a max total size.
    """
    parts = [format_doc_for_context(d) for d in docs]
    context = "\n\n---\n\n".join(parts)
    if len(context) > MAX_CONTEXT_CHARS:
        # gradually trim documents (drop last ones) until under limit
        truncated = []
        curr = 0
        for p in parts:
            if curr + len(p) + 6 > MAX_CONTEXT_CHARS:
                break
            truncated.append(p)
            curr += len(p) + 6
        context = "\n\n---\n\n".join(truncated)
    return context

# Retrieval helpers
def hybrid_retrieve(query: str):
    """
    Return a list of (Document, score_or_none) in ranked order.
    Prefer using similarity_search_with_score when available to obtain scores.
    Fallback to retriever.get_relevant_documents() and docsearch.similarity_search() if needed.
    """
    docs_with_scores = []

    # 1) Try similarity_search_with_score which returns (doc, score)
    try:
        # many langchain-pinecone wrappers expose similarity_search_with_score
        sim_ret = getattr(vs, "similarity_search_with_score", None)
        if callable(sim_ret):
            pairs = vs.similarity_search_with_score(query, k=SIM_SEARCH_K)
            for item in pairs:
                # item may be (doc, score) pair
                if isinstance(item, tuple) and len(item) >= 2:
                    d, s = item[0], item[1]
                    docs_with_scores.append((d, float(s)))
    except Exception:
        # ignore and fall back below
        docs_with_scores = []

    # 2) Also get retriever semantic docs (no scores)
    try:
        sem_docs = retriever.get_relevant_documents(query)
    except Exception:
        sem_docs = []

    # 3) If we didn't get scores, attempt a similarity_search (no scores) and stitch with sem_docs
    if not docs_with_scores:
        try:
            maybe_kw = vs.similarity_search(query, k=SIM_SEARCH_K)
        except Exception:
            maybe_kw = []
        # combine sem_docs and maybe_kw, dedupe
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
        # docs_with_scores exists; also ensure sem_docs added (but keep top scores order)
        # dedupe by content
        seen = set()
        final = []
        # Add scored first (sorted descending by score)
        docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        for d, s in docs_with_scores:
            key = (getattr(d, "page_content", "") or "")[:400]
            if key and key not in seen:
                seen.add(key)
                final.append((d, s))
        # now add sem_docs (no score) if not present
        for d in sem_docs:
            key = (getattr(d, "page_content", "") or "")[:400]
            if key and key not in seen:
                seen.add(key)
                final.append((d, None))
        return final

# RAG generation
def rag_generate(query: str, doc_score_pairs):
    """
    Compose context, decide if there is enough evidence, then call LLM with structured system prompt.
    doc_score_pairs: list of (Document, score_or_none)
    """
    if not doc_score_pairs:
        return "I don't know based on the provided Reddit data."

    # compute simple evidence metric:
    numeric_scores = [s for (_, s) in doc_score_pairs if s is not None]
    avg_score = float(sum(numeric_scores) / len(numeric_scores)) if numeric_scores else None
    num_docs = len(doc_score_pairs)

    # Conservative rule: only fallback if there are very few docs OR numeric scores exist and are all very low
    if num_docs < 1:
        return "I don't know based on the provided Reddit data."
    if numeric_scores:
        if avg_score < EVIDENCE_SCORE_THRESH and num_docs < MIN_DOCS_FOR_CONFIDENCE:
            # not enough evidence
            return "I don't know based on the provided Reddit data."

    # Build docs list for context (convert to Documents only)
    docs_only = [d for (d, _) in doc_score_pairs]

    # prepare context (formatted)
    context = build_context(docs_only)
    # collect subreddits used for a quick citation hint
    subreddits = sorted({(d.metadata or {}).get("subreddit", "") for d in docs_only if (d.metadata or {}).get("subreddit")})
    sub_list = ", ".join(subreddits) if subreddits else "unknown"

    # system promptt
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
    except Exception as e:
        # fallback simple summarization attempt using only the joined excerpts
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

# Relevance scoring (LLM)
def relevance_score(question: str, answer: str):
    prompt = (
        "On a scale 1–5, how relevant is the ANSWER to the QUESTION?\n"
        "Return only a single integer 1,2,3,4 or 5.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
    )
    try:
        res = llm.invoke(prompt)
        return res.content.strip()
    except Exception:
        return "0"

# Driver
def run_evaluation():
    questions = load_questions()
    results = []

    for q in questions:
        query = q.get("question") or q.get("q") or ""
        print(f"\nEvaluating: {query}")

        doc_score_pairs = hybrid_retrieve(query)  # list[(Document, score_or_none)]
        docs_used = len(doc_score_pairs)

        rag_ans = rag_generate(query, doc_score_pairs)
        llm_ans = llm.invoke(query).content.strip()

        rag_rel = relevance_score(query, rag_ans)
        llm_rel = relevance_score(query, llm_ans)

        results.append({
            "question": query,
            "rag_answer": rag_ans,
            "llm_answer": llm_ans,
            "rag_relevance": rag_rel,
            "llm_relevance": llm_rel,
            "documents_used": docs_used,
            "avg_evidence_score": (sum([s for (_, s) in doc_score_pairs if s is not None]) / len([s for (_, s) in doc_score_pairs if s is not None])) if any([s is not None for (_, s) in doc_score_pairs]) else None
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✔ Evaluation complete: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()

