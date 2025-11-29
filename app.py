# render_app/app.py
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "reddit-insights")

app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize Pinecone client
pc = None
docsearch = None
retriever = None

try:
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    idx_list_resp = pc.list_indexes()
    index_list = [i.get("name") for i in idx_list_resp if i.get("name")]
    if INDEX_NAME in index_list:
        # Init LangChain store if needed
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        try:
            docsearch = PineconeVectorStore.from_existing_index(
                index_name=INDEX_NAME,
                embedding=embeddings
            )
            retriever = docsearch.as_retriever(search_kwargs={"k": 8})
        except Exception as e:
            print("Warning: Could not init PineconeVectorStore:", e)
    else:
        print(f"Index '{INDEX_NAME}' not found in Pinecone indexes:", index_list)

except Exception as e:
    print("Error initializing Pinecone client:", e)
    pc = None

# LLM / prompt
system_prompt = (
    "You are a research assistant summarizing Reddit discussions about software tools "
    "used in Law, Construction, and Tech industries.\n\n"
    "Use ONLY the provided Reddit excerpts to answer accurately. "
    "If the context does not include relevant data, respond: "
    "'I don’t know based on the provided Reddit data.'\n\n"
    "Context:\n{context}"
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/chat")
def chat_page():
    return render_template("chatbot.html")

@app.route("/api/pinecone_stats")
def pinecone_stats():
    """Return stats for dashboard — vector count, dimension, etc."""
    if pc is None:
        return jsonify({"status": "error", "error": "Pinecone client not initialized"}), 500

    try:
        idx = pc.Index(INDEX_NAME)
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"Could not open index: {repr(e)}"
        }), 500

    try:
        stats = idx.describe_index_stats()
    except Exception as e:
        # fallback: index exists but cannot fetch stats
        return jsonify({
            "status": "connected",
            "index_exists": True,
            "error": f"Could not fetch stats: {repr(e)}"
        }), 200

    # Build response
    return jsonify({
        "status": "ok",
        "index_exists": True,
        "total_vectors": stats.get("total_vector_count"),
        "namespaces": stats.get("namespaces"),
        "dimension": stats.get("dimension"),
        "metric": stats.get("metric"),
        "raw_stats": stats
    }), 200

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "No message provided", 400
    if not docsearch or not retriever:
        return "Index not ready", 503

    try:
        docs = retriever.get_relevant_documents(msg)
    except Exception:
        # fallback search
        retrieved = docsearch.similarity_search_with_score(msg, k=8)
        docs = [d for d, _ in retrieved]

    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        return "I don’t know based on the provided Reddit data."

    response = question_answer_chain.invoke({"input": msg, "context": docs})
    if isinstance(response, str):
        return response.strip()
    return str(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)

