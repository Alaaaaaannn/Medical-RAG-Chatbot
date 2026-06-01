import os
from typing import List, Optional

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src import config, prompter


def build_rag_chain(llm, retriever):
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", prompter.contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", prompter.system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def check_emergency(text: str) -> Optional[str]:
    low = text.lower()
    if any(k in low for k in prompter.EMERGENCY_KEYWORDS):
        return prompter.EMERGENCY_MESSAGE
    return None


def format_sources(context) -> List[dict]:
    seen, sources = set(), []
    for d in context:
        meta = d.metadata or {}
        src = os.path.basename(str(meta.get("source", "unknown")))
        page = meta.get("page")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": src, "page": page, "snippet": d.page_content[:180].strip()})
    return sources


def verify_grounding(llm, answer: str, context) -> bool:
    ctx = "\n\n".join(d.page_content for d in context)
    prompt = (
        "You are a strict fact-checker. Based ONLY on the CONTEXT, is the ANSWER fully "
        "supported? Reply with exactly 'yes' or 'no'.\n\nCONTEXT:\n" + ctx +
        "\n\nANSWER:\n" + answer
    )
    try:
        return llm.invoke(prompt).content.strip().lower().startswith("yes")
    except Exception:
        return True 


def answer_question(rag_chain, llm, question: str, chat_history=None, budget=None) -> dict:
    chat_history = chat_history or []

    emergency = check_emergency(question)
    if emergency:
        return {"answer": emergency, "sources": [], "emergency": True}
    if budget is not None and not budget.allow():
        return {"answer": prompter.DAILY_LIMIT_MESSAGE, "sources": [],
                "emergency": False, "limited": True}

    result = rag_chain.invoke({"input": question, "chat_history": chat_history})
    answer = result.get("answer", "")
    context = result.get("context", [])

    if not context:
        answer = prompter.NO_CONTEXT_MESSAGE
    elif config.GROUNDING_CHECK and not verify_grounding(llm, answer, context):
        answer = answer + "\n\n" + prompter.GROUNDING_CAUTION

    return {"answer": answer, "sources": format_sources(context), "emergency": False}
