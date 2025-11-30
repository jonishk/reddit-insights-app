# render_app/app.py

import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# LangChain + Pinecone + OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from pinecone import Pinecone
import pinecone

load_dotenv()

# Environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "reddit-insights")

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Global variables
pc = None
docsearch = None
retriever = None

print("=== Render Init ===")
print("Pinecone SDK Version:", pinecone.__version__)
print("Using index:", INDEX_NAME)
print("====================")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_list = [i["name"] for i in pc.list_indexes()]

    if INDEX_NAME not in index_list:
        print(f"[WARN] Pinecone index '{INDEX_NAME}' not found. Chatbot will not work until pipeline uploads embeddings.")
    else:
        print("[OK] Pinecone index found.")
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings
        )
        retriever = docsearch.as_retriever(search_kwargs={"k": 8})

except Exception as e:
    print("Failed to initialize Pinecone:", e)
    pc = None
    docsearch = None
    retriever = None


# LLM + prompt
system_prompt = (
    "You are a research assistant summarizing Reddit discussions about software tools "
    "used in Law, Construction, and Tech industries.\n\n"
    "Use ONLY the provided Reddit excerpts to answer accurately. "
    "If the context does not include relevant data, respond:\n"
    "'I don't know based on the provided Reddit data.'\n\n"
    "Context:\n{context}"
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)


# ----------------------------- ROUTES -------------------------------- #

@app.route("/")
def chat_page():
    return render_template("chatbot.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()

    if not msg:
        return "Please enter a message."

    if docsearch is None or retriever is None:
        return "Pinecone index not ready. Please run the local pipeline first."

    # category filter (optional)
    text = msg.lower()
    search_filter = None
    if "construction" in text:
        search_filter = {"category": {"$eq": "Construction"}}
    elif "law" in text or "legal" in text:
        search_filter = {"category": {"$eq": "Law"}}
    elif "tech" in text or "software" in text:
        search_filter = {"category": {"$eq": "Tech"}}

    try:
        docs = retriever.get_relevant_documents(msg)
    except Exception:
        # fallback method
        retrieved = docsearch.similarity_search_with_score(msg, k=8, filter=search_filter)
        docs = [d for d, s in retrieved]

    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not docs:
        return "I don't know based on the provided Reddit data."

    try:
        response = question_answer_chain.invoke({"input": msg, "context": docs})
        answer = response.strip() if isinstance(response, str) else str(response)
        return answer if answer else "I don't know based on the provided Reddit data."
    except Exception as e:
        return f"Error generating answer: {e}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
