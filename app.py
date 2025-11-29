# render_app/app.py
import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# LangChain + Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from pinecone import Pinecone


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "reddit-insights")

app = Flask(__name__, static_folder="static", template_folder="templates")

# ------------------------------
# OpenAI Embeddings
# ------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ------------------------------
# Pinecone init (v6.x)
# ------------------------------
pc = None
docsearch = None
retriever = None

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Pinecone 6.x returns {"indexes": [{ "name": ... }]}
    index_data = pc.list_indexes()
    existing_indexes = [i["name"] for i in index_data.get("indexes", [])]

    if INDEX_NAME not in existing_indexes:
        print(f"Index '{INDEX_NAME}' not found in Pinecone.")
    else:
        print("Connected to Pinecone index.")

        # LangChain vector store uses its own wrapper
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings
        )
        retriever = docsearch.as_retriever(search_kwargs={"k": 8})

except Exception as e:
    print("Could not initialize Pinecone:", e)
    pc = None
    docsearch = None
    retriever = None

# ------------------------------
# LLM
# ------------------------------
system_prompt = (
    "You are a research assistant summarizing Reddit discussions.\n"
    "Use the provided context. If no relevant context exists, reply:\n"
    "'I don’t know based on the provided Reddit data.'\n\n"
    "Context:\n{context}"
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# ------------------------------
# Routes
# ------------------------------

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/chat")
def chat_page():
    return render_template("chatbot.html")


# ------------------------------
# API: Pinecone Index Stats
# ------------------------------
@app.route("/api/pinecone_stats")
def pinecone_stats():
    if pc is None:
        return jsonify({"error": "Pinecone client not initialized"}), 503

    try:
        # List indexes safely
        index_data = pc.list_indexes()
        existing = [i["name"] for i in index_data.get("indexes", [])]

        if INDEX_NAME not in existing:
            return jsonify({
                "index_exists": False,
                "message": f"Index '{INDEX_NAME}' not found. Run local pipeline."
            })

        # Load index handle (Pinecone v6)
        try:
            idx = pc.Index(INDEX_NAME)
        except Exception as e:
            return jsonify({"error": f"Index init error: {str(e)}"}), 500

        # Describe stats (v6 returns dict)
        try:
            stats = idx.describe_index_stats()
        except Exception as e:
            return jsonify({"error": f"Stats fetch error: {str(e)}"}), 500

        namespaces = list(stats.get("namespaces", {}).keys())
        total_vectors = stats.get("total_vector_count", 0)
        dimension = stats.get("dimension", 1536)

        return jsonify({
            "index_exists": True,
            "total_vectors": total_vectors,
            "dimension": dimension,
            "metric": stats.get("metric", "cosine"),
            "namespaces": namespaces,
            "raw_stats": stats
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ------------------------------
# Chat endpoint
# ------------------------------
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please send a message.", 400

    if docsearch is None or retriever is None:
        return "Pinecone index not ready. Run local pipeline to upload embeddings.", 503

    # Category filter
    text = msg.lower()
    search_filter = None
    if "construction" in text:
        search_filter = {"category": {"$eq": "Construction"}}
    elif "law" in text or "legal" in text:
        search_filter = {"category": {"$eq": "Law"}}
    elif "tech" in text or "software" in text:
        search_filter = {"category": {"$eq": "Tech"}}

    try:
        try:
            docs = retriever.get_relevant_documents(msg)
        except Exception:
            retrieved = docsearch.similarity_search_with_score(msg, k=8, filter=search_filter)
            docs = [d for d, _ in retrieved]

    except Exception as e:
        return f"Error during retrieval: {e}", 500

    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        return "I don’t know based on the provided Reddit data."

    try:
        response = question_answer_chain.invoke({"input": msg, "context": docs})
        answer = response.strip() if isinstance(response, str) else str(response)
        return answer or "I don’t know based on the provided Reddit data."
    except Exception as e:
        return f"Error generating answer: {e}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
