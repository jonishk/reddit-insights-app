import os
import sys
import io
import json
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# UTF-8 and setup
sys.stdout.reconfigure(line_buffering=True)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")


# Pinecone + embeddings
print(" Connecting to Pinecone and loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
index_name = "reddit-insights"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)


# LLM setup
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=400)

system_prompt = (
    "You are a research assistant summarizing Reddit discussions about software tools "
    "used in Law, Construction, and Tech industries.\n\n"
    "Use the Reddit excerpts below to answer accurately. You may make brief, logical inferences from the context but avoid unsupported assumptions.\n"
    "If the context does not include relevant data, respond with:\n"
    "'I don’t know based on the provided Reddit data.'\n\n"
    "Include subreddit or profession context if available.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)


# Load test questions
try:
    with open("questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
except FileNotFoundError:
    print(" questions.json not found — please create it with test questions.")
    sys.exit(1)


# Helper functions
def get_search_filter(msg):
    """Detect domain keywords for category filtering."""
    if "construction" in msg.lower():
        return {"category": "Construction"}
    elif "law" in msg.lower() or "legal" in msg.lower():
        return {"category": "Law"}
    elif "tech" in msg.lower() or "software" in msg.lower():
        return {"category": "Tech"}
    else:
        return None

def rag_answer(question):
    """Retrieve docs and generate RAG response."""
    search_filter = get_search_filter(question)
    retrieved_docs_with_scores = docsearch.similarity_search_with_score(question, k=10, filter=search_filter)
    relevant_docs = [doc for doc, score in retrieved_docs_with_scores if doc.page_content.strip()]

    if not relevant_docs:
        return "I don’t know based on the provided Reddit data.", []

    docs_for_chain = [Document(page_content=d.page_content, metadata=d.metadata or {}) for d in relevant_docs]
    response = question_answer_chain.invoke({"input": question, "context": docs_for_chain})
    answer = response.strip() if isinstance(response, str) else getattr(response, "content", str(response))
    return answer, relevant_docs

def llm_only_answer(question):
    """Generate LLM-only answer."""
    response = llm.invoke(question)
    return response.content.strip() if hasattr(response, "content") else str(response)

def compute_relevance(answer, expected_source):
    """Basic binary relevance scoring."""
    if "don't know" in answer.lower():
        return 0
    elif expected_source.lower() == "out-of-scope":
        return 0
    else:
        return 1


# Evaluation loop
results = []
print("\n🔹 Running evaluation on question set...\n")

for q in questions:
    question = q["question"]
    expected_source = q.get("expected_source", "Unknown")

    print(f"Evaluating: {question[:70]}...")
    rag_resp, retrieved = rag_answer(question)
    llm_resp = llm_only_answer(question)

    rag_rel = compute_relevance(rag_resp, expected_source)
    llm_rel = compute_relevance(llm_resp, expected_source)

    results.append({
        "question": question,
        "expected_source": expected_source,
        "rag_answer": rag_resp,
        "llm_answer": llm_resp,
        "rag_relevance": rag_rel,
        "llm_relevance": llm_rel
    })


# Save + Metrics
df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False, encoding="utf-8")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Helper functions
def compute_relevance(answer: str, expected_source: str):
    """
    Assigns a relevance score:
    - 1 if the model correctly identifies scope and gives a contextual answer
    - 0 if out-of-scope but model attempts to answer
    - 0 if in-scope but model says 'I don’t know'
    - 0.5 if answer is vague (heuristic)
    """
    ans_lower = answer.lower()
    if "i don’t know" in ans_lower or "i don't know" in ans_lower:
        return 0

    if expected_source.lower() == "out-of-scope":
        # Should say "I don't know" for out-of-scope questions
        return 0

    vague_terms = ["maybe", "possibly", "could", "might", "unclear", "not sure"]
    if any(v in ans_lower for v in vague_terms):
        return 0.5

    # Otherwise treat as relevant
    return 1


# Evaluation loop
results = []
y_true, y_pred_rag, y_pred_llm = [], [], []

print("Running evaluation on question set...")
for q in questions:
    question = q["question"]
    expected_source = q.get("expected_source", "Unknown")

    print(f"\nEvaluating: {question[:80]}...")
    rag_resp, retrieved = rag_answer(question)
    llm_resp = llm_only_answer(question)

    rag_rel = compute_relevance(rag_resp, expected_source)
    llm_rel = compute_relevance(llm_resp, expected_source)

    # Ground truth (1 = in-scope, 0 = out-of-scope)
    y_true.append(1 if expected_source.lower() == "in-scope" else 0)
    y_pred_rag.append(1 if rag_rel > 0 else 0)
    y_pred_llm.append(1 if llm_rel > 0 else 0)

    results.append({
        "question": question,
        "expected_source": expected_source,
        "rag_answer": rag_resp,
        "llm_answer": llm_resp,
        "rag_relevance": rag_rel,
        "llm_relevance": llm_rel
    })


# Save results
df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False, encoding="utf-8")


# Compute Metrics
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

rag_metrics = compute_metrics(y_true, y_pred_rag)
llm_metrics = compute_metrics(y_true, y_pred_llm)

print("\n Evaluation complete! Saved to evaluation_results.csv")
print(f" RAG Metrics:\nAccuracy: {rag_metrics['accuracy']:.2f}\nPrecision: {rag_metrics['precision']:.2f}\nRecall: {rag_metrics['recall']:.2f}\nF1: {rag_metrics['f1']:.2f}")
print(f"\n LLM Metrics:\nAccuracy: {llm_metrics['accuracy']:.2f}\nPrecision: {llm_metrics['precision']:.2f}\nRecall: {llm_metrics['recall']:.2f}\nF1: {llm_metrics['f1']:.2f}")


