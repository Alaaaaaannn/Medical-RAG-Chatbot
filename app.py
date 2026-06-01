import uuid

from flask import Flask, render_template, request, jsonify, session
from werkzeug.exceptions import RequestEntityTooLarge
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from src import config, helper
from src.cache import DailyBudget, SessionStore
from src.retriever import build_retriever
from src.chain import build_rag_chain, answer_question

app = Flask(__name__)
app.secret_key = config.FLASK_SECRET
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

TOO_LONG_MESSAGE = (
    f"Your message is too long (max {config.MAX_INPUT_CHARS} characters). "
    "Please ask a shorter, more focused question."
)
print("Initialising Medical RAG Chatbot ...")
embeddings = helper.embeddings_init()
docsearch = PineconeVectorStore.from_existing_index(
    index_name=config.INDEX_NAME, embedding=embeddings
)
chunks = helper.load_chunks()  
if chunks is None:
    print("  (no cached chunks found -> hybrid BM25 disabled; run store_index.py)")

llm = ChatGroq(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE,
    max_tokens=config.LLM_MAX_TOKENS,  
)
retriever = build_retriever(docsearch, chunks, llm)
rag_chain = build_rag_chain(llm, retriever)

budget = DailyBudget(config.DAILY_BUDGET)
sessions = SessionStore(config.MAX_SESSIONS, config.SESSION_TTL_SECONDS)
print(f"Ready. Daily budget: {config.DAILY_BUDGET} requests, max_tokens: {config.LLM_MAX_TOKENS}.")


def _remember(history: list, question: str, answer: str) -> None:
    history.extend([HumanMessage(content=question), AIMessage(content=answer)])
    if len(history) > config.MAX_HISTORY_MESSAGES:
        del history[: -config.MAX_HISTORY_MESSAGES]


@app.errorhandler(RequestEntityTooLarge)
def _too_large(_e):
    return jsonify({"answer": TOO_LONG_MESSAGE, "sources": []}), 413


@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def get_response():
    msg = (request.form.get("msg") or "").strip()
    if not msg:
        return jsonify({"answer": "Please enter a question.", "sources": []}), 400
    if len(msg) > config.MAX_INPUT_CHARS:
        return jsonify({"answer": TOO_LONG_MESSAGE, "sources": []}), 413

    history = sessions.get(session.get("sid", "anon"))
    result = answer_question(rag_chain, llm, msg, history, budget)
    if not result.get("emergency") and not result.get("limited"):
        _remember(history, msg, result["answer"])
    return jsonify(result)


@app.route("/reset", methods=["POST"])
def reset():
    sessions.reset(session.get("sid", "anon"))
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
