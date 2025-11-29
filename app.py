# render_app/app.py
import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import pinecone

# LangChain + Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# ===============================
# ENV + INIT
# ===============================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "reddit-insights")

if not PINECONE_API_KEY:
    print("WARNING: Missing PINECONE_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: Missing OPENAI_API_KEY")

app = Flask(__name__, static_folder="static", template_folder="templates")


# ===============================
# PINECONE v7 CLIENT INIT
# ===============================
pc = None
docsearch = None
retriever = None

try:
    print("Pinecone SDK Version:", pinecone.__version__)
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

    # List indexes
    index_list = [i["name"] for i in pc.list_indexes()]
    print("Pinecone Indexes:", index_list)

    if INDEX_NAME not in index_list:
        print(f"Index '{INDEX_NAME}' not found. App will still run but needs local pipeline upload.")
    else:
        print(f"Pinecone index '{INDEX_NAME}' is available")

        # LangChain vectorstore
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings
        )
        retriever = docsearch.as_retriever(search_kwargs={"k": 8})
        print("VectorStore + Retriever ready")

except Exception as e:
    print("Pinecone initialization failed:", e)
    pc = None
    docsearch = None
    retriever = None


# ===============================
# LLM + PROMPT
# ===============================
system_prompt = (
    "You are a research assistant summarizing Reddit discussions about software tools "
    "used in Law, Construction, and Tech industries.\n\n"
    "Use the provided Reddit excerpts to answer accurately. If the context does not "
    "include relevant data, reply:\n'I don’t know based on the provided Reddit data.'\n\n"
    "Context:\n{context}"
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=400)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)


# ===============================
# ROUTES
# ===============================
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/chat")
def chat_page():
    return render_template("chatbot.html")


# ===============================
# PINECONE STATS API (FIXED)
# ===============================
@app.route("/api/pinecone_stats")
def pinecone_stats():
    if pc is None:
        return jsonify({"error": "Pinecone client not initialized"}), 500

    try:
        # list indexes
        try:
            index_list = [i["name"] for i in pc.list_indexes()]
        except Exception as e:
            return jsonify({
                "error": f"Unable to list indexes: {str(e)}"
            }), 500

        # index absent
        if INDEX_NAME not in index_list:
            return jsonify({
                "index_exists": False,
                "message": f"Index '{INDEX_NAME}' not found. Run local pipeline upload."
            })

        # ******** THE FIX: use pc.Index(), not pc.index() ********
        try:
            idx = pc.Index(INDEX_NAME)
        except Exception as e:
            return jsonify({
                "error": f"Could not connect to index: {repr(e)}"
            }), 500

        # get stats
        try:
            stats = idx.describe_index_stats()
        except Exception as e:
            return jsonify({
                "error": f"describe_index_stats failed: {repr(e)}"
            }), 500

        return jsonify({
            "index_exists": True,
            "total_vectors": stats.get("total_vector_count"),
            "dimension": stats.get("dimension"),
            "metric": stats.get("metric"),
            "namespaces": list(stats.get("namespaces", {}).keys()),
            "raw_stats": stats
        })

    except Exception as e:
        return jsonify({"error": repr(e)}), 500



# ===============================
# CHATBOT ENDPOINT
# ===============================
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please send a message.", 400

    if not docsearch or not retriever:
        return "Pinecone index not ready. Run local pipeline to upload embeddings.", 503

    # Category filter (optional)
    txt = msg.lower()
    search_filter = None
    if "construction" in txt:
        search_filter = {"category": {"$eq": "Construction"}}
    elif "law" in txt or "legal" in txt:
        search_filter = {"category": {"$eq": "Law"}}
    elif "tech" in txt or "software" in txt:
        search_filter = {"category": {"$eq": "Tech"}}

    # Retrieval
    try:
        try:
            docs = retriever.get_relevant_documents(msg)
        except Exception:
            retrieved = docsearch.similarity_search_with_score(msg, k=8, filter=search_filter)
            docs = [d for d, s in retrieved]
    except Exception as e:
        return f"Error during retrieval: {e}", 500

    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        return "I don’t know based on the provided Reddit data."

    # Generate answer
    try:
        response = question_answer_chain.invoke({"input": msg, "context": docs})
        answer = response if isinstance(response, str) else str(response)
        return answer or "I don’t know based on the provided Reddit data."
    except Exception as e:
        return f"Error while generating answer: {e}", 500


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)

