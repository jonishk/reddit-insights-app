# render_app/app.py
import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# -----------------------------
# DEBUG OUTPUT (SAFE)
# -----------------------------
print("===== RENDER DEBUG START =====")

import pinecone
print("Pinecone SDK Version:", getattr(pinecone, "__version__", "UNKNOWN"))
print("Pinecone Module:", pinecone)

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "reddit-insights")

print("Has OPENAI key:", bool(OPENAI_API_KEY))
print("Has PINECONE key:", bool(PINECONE_API_KEY))
print("Using index name:", INDEX_NAME)

# -----------------------------
# Flask
# -----------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

# -----------------------------
# Pinecone Debug Initialization
# -----------------------------
pc = None
docsearch = None
retriever = None

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("\n===== INITIALIZING PINECONE CLIENT =====")

try:
    from pinecone import Pinecone  # v6 client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone client object:", pc)

    # List indexes
    try:
        index_data = pc.list_indexes()
        print("List Indexes RETURNED:", index_data)
        index_names = [i["name"] for i in index_data]
    except Exception as e:
        print("ERROR listing indexes:", repr(e))
        index_names = []

    if INDEX_NAME not in index_names:
        print(f"Index '{INDEX_NAME}' NOT found.")
    else:
        print(f"Index '{INDEX_NAME}' FOUND.")

        # Attempt to load LangChain vector store
        try:
            print("Creating LangChain PineconeVectorStore...")
            docsearch = PineconeVectorStore.from_existing_index(
                index_name=INDEX_NAME,
                embedding=embeddings
            )
            retriever = docsearch.as_retriever(search_kwargs={"k": 8})
            print("VectorStore + Retriever READY.")
        except Exception as e:
            print("ERROR initializing PineconeVectorStore:", repr(e))

except Exception as e:
    print("CRITICAL ERROR initializing Pinecone CLIENT:", repr(e))
    pc = None

print("===== PINECONE INIT COMPLETE =====\n")


# -----------------------------------------------------
#  LLM + Prompt
# -----------------------------------------------------
system_prompt = (
    "You are a research assistant summarizing Reddit discussions about software tools "
    "in Law, Construction, and Tech industries.\n\n"
    "If context lacks relevant data, respond:\n"
    "'I don’t know based on the provided Reddit data.'\n\n"
    "Context:\n{context}"
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=400)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)


# -----------------------------------------------------
# Routes
# -----------------------------------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/chat")
def chat_page():
    return render_template("chatbot.html")


@app.route("/api/pinecone_stats")
def pinecone_stats():
    """Report exact Pinecone index stats to dashboard."""
    if pc is None:
        return jsonify({"error": "Pinecone client failed to initialize"}), 500

    try:
        index_list = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME not in index_list:
            return jsonify({
                "index_exists": False,
                "message": f"Index '{INDEX_NAME}' not found."
            })

        # IMPORTANT: v6 client uses pc.Index()
        print("Creating index handle via pc.Index(...)")
        try:
            idx = pc.Index(INDEX_NAME)
        except Exception as e:
            return jsonify({"error": f"Unable to open index: {repr(e)}"}), 500

        try:
            stats = idx.describe_index_stats()
        except Exception as e:
            return jsonify({"error": f"describe_index_stats failed: {repr(e)}"}), 500

        return jsonify({
            "index_exists": True,
            "raw_stats": stats,
            "total_vectors": stats.get("total_vector_count", None),
            "dimension": stats.get("dimension", None),
            "metric": stats.get("metric", None),
            "namespaces": list(stats.get("namespaces", {}).keys())
        })

    except Exception as e:
        return jsonify({"error": repr(e)}), 500


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please send a message.", 400

    if not docsearch or not retriever:
        return "Pinecone index not ready.", 503

    # Retrieve docs
    try:
        docs = retriever.get_relevant_documents(msg)
    except Exception:
        retrieved = docsearch.similarity_search_with_score(msg, k=8)
        docs = [d for d, s in retrieved]

    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        return "I don’t know based on the provided Reddit data."

    try:
        result = question_answer_chain.invoke({"input": msg, "context": docs})
        return result.strip() if isinstance(result, str) else str(result)
    except Exception as e:
        return f"Error while generating answer: {repr(e)}", 500


if __name__ == "__main__":
    print("Starting Flask on Render...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
