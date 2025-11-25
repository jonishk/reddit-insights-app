# render_app/app.py
import os
import time
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# LangChain + Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure keys are set
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY or PINECONE_API_KEY missing in environment.")

# Flask
app = Flask(__name__, static_folder="static", template_folder="templates")

# RAG / embeddings (OpenAI text-embedding-3-small => 1536 dims)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = os.getenv("PINECONE_INDEX_NAME", "reddit-insights")

pc = None
docsearch = None
retriever = None

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing:
        # If index doesn't exist yet, we still continue — pipeline should create it locally.
        print(f"Pinecone index '{index_name}' not found. Render app will still run but index must be created/uploaded by local pipeline.")
    else:
        print("Connected to Pinecone index.")
        docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
        retriever = docsearch.as_retriever(search_kwargs={"k": 8})
except Exception as e:
    print("Could not initialize Pinecone (Render app). This is expected if index not yet created. Error:", e)
    pc = None
    docsearch = None
    retriever = None

# LLM / prompt
system_prompt = (
    "You are a research assistant summarizing Reddit discussions about software tools "
    "used in Law, Construction, and Tech industries.\n\n"
    "Use the provided Reddit excerpts to answer accurately. If the context does not include "
    "relevant data, respond: 'I don’t know based on the provided Reddit data.'\n\n"
    "Context:\n{context}"
)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=400)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Routes
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/chat")
def chat_page():
    return render_template("chatbot.html")


@app.route("/api/pinecone_stats")
def pinecone_stats():
    """Return index stats for dashboard (safe for Render)."""

    # 1. Verify Pinecone client exists
    if pc is None:
        return jsonify({
            "error": "Pinecone client not initialized. Check PINECONE_API_KEY."
        }), 503

    try:
        # 2. Check connection / list indexes
        try:
            index_list = [i["name"] for i in pc.list_indexes()]
        except Exception as e:
            return jsonify({
                "error": f"Unable to list indexes: {str(e)}",
                "index_exists": False
            }), 500

        # 3. Index not found
        if index_name not in index_list:
            return jsonify({
                "index_exists": False,
                "message": f"Index '{index_name}' not found. "
                           "Run local pipeline to create & upload embeddings."
            })

        # 4. Safely initialize index handle using new API
        try:
            idx = pc.Index(index_name)
        except Exception as e:
            return jsonify({
                "error": f"Could not initialize Pinecone Index object: {str(e)}"
            }), 500

        # 5. Get stats safely
        try:
            stats = idx.describe_index_stats()
        except Exception as e:
            return jsonify({
                "error": f"Could not retrieve index stats: {str(e)}"
            }), 500

        # 6. Extract fields
        namespaces = list(stats.get("namespaces", {}).keys())
        total_vectors = stats.get("total_vector_count", 0)
        dimension = stats.get("dimension", 1536)
        metric = stats.get("metric", "cosine")

        return jsonify({
            "index_exists": True,
            "total_vectors": total_vectors,
            "dimension": dimension,
            "metric": metric,
            "namespaces": namespaces,
            "raw_stats": stats
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get", methods=["POST"])
def chat():
    """Chat endpoint used by UI. Only retrieval + LLM generation (no pipeline)."""
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please send a message.", 400

    if not docsearch or not retriever:
        return "Pinecone index not ready. Please run local pipeline to upload embeddings.", 503

    # Simple category filter heuristics (optional)
    text = msg.lower()
    search_filter = None
    if "construction" in text:
        search_filter = {"category": {"$eq": "Construction"}}
    elif "law" in text or "legal" in text:
        search_filter = {"category": {"$eq": "Law"}}
    elif "tech" in text or "software" in text:
        search_filter = {"category": {"$eq": "Tech"}}

    try:
        # Use the retriever to get relevant docs
        # If the underlying langchain/pinecone integration supports filter param in retriever, pass it.
        # We'll fallback to docsearch.similarity_search_with_score if needed.
        try:
            # newer style: retriever.get_relevant_documents
            docs = retriever.get_relevant_documents(msg)
        except Exception:
            # fallback
            retrieved = docsearch.similarity_search_with_score(msg, k=8, filter=search_filter)
            docs = [d for d, s in retrieved]
    except Exception as e:
        return f"Error during retrieval: {e}", 500

    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not docs:
        return "I don’t know based on the provided Reddit data."

    try:
        response = question_answer_chain.invoke({"input": msg, "context": docs})
        answer = response.strip() if isinstance(response, str) else str(response)
        if not answer:
            return "I don’t know based on the provided Reddit data."
        return answer
    except Exception as e:
        return f"Error while generating answer: {e}", 500


if __name__ == "__main__":
    # When running locally for development, run Flask directly.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)


