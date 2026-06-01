system_prompt = (
    "You are a careful medical information assistant for educational use.\n"
    "Use ONLY the retrieved context below to answer the user's question.\n\n"
    "Rules:\n"
    "- If the answer is not in the context, say you don't know and recommend consulting a "
    "licensed healthcare professional. Never invent facts.\n"
    "- Provide general information only. Do NOT diagnose, prescribe, or give personalised "
    "dosage instructions.\n"
    "- Always remind the user to consult a qualified healthcare professional for personal "
    "medical concerns.\n"
    "- Be concise (max ~5 sentences).\n\n"
    "Context:\n"
    "{context}"
)

contextualize_q_system_prompt = (
    "Given the chat history and the latest user question which might reference the chat "
    "history, formulate a standalone question that can be understood without the chat "
    "history. Do NOT answer it; just reformulate it if needed, otherwise return it unchanged."
)

EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "suicidal", "suicide", "kill myself", "want to die", "self harm", "self-harm",
    "overdose", "unconscious", "stroke", "heart attack", "severe bleeding",
    "anaphylaxis", "seizure",
]
EMERGENCY_MESSAGE = (
    "This may be a medical emergency. I cannot help with urgent situations. "
    "Please call your local emergency number immediately (e.g. 911 in the US, 112 in the EU, "
    "108 in India) or go to the nearest emergency department. If you are in crisis, contact a "
    "local crisis hotline now."
)

NO_CONTEXT_MESSAGE = (
    "I couldn't find information about that in my medical knowledge base, so I can't answer it "
    "reliably. Please consult a licensed healthcare professional."
)
GROUNDING_CAUTION = (
    "Note: parts of this answer may not be fully supported by my source documents. "
    "Please verify with a licensed healthcare professional."
)

DAILY_LIMIT_MESSAGE = (
    "This assistant has reached its daily usage limit and is temporarily unavailable. "
    "Please try again tomorrow."
)