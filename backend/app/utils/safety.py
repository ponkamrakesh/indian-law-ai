import re
from typing import Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings

# Simple keyword-based harmful intent detection (expand as needed)
HARMFUL_KEYWORDS = [
    "how to commit murder", "how to hack", "illegal drug manufacturing",
    "evade taxes illegally", "bribe a judge", "forge documents",
    "kidnap someone", "terrorist attack", "how to launder money"
]

def is_harmful_query(query: str) -> bool:
    """Basic safety check for harmful/criminal intent queries."""
    query_lower = query.lower()
    for kw in HARMFUL_KEYWORDS:
        if kw in query_lower:
            return True
    # Additional regex for suspicious patterns
    if re.search(r'\b(how to|steps to|guide to)\s+(kill|murder|steal|fraud|crime)\b', query_lower):
        return True
    return False

def llm_safety_check(query: str) -> Tuple[bool, str]:
    """Use LLM for deeper safety/guardrail check. Returns (is_safe, reason)."""
    llm = ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.llm_model,
        temperature=0.0,
        max_tokens=200
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict safety guardrail for a legal AI assistant. "
                   "Determine if the user query is asking for help with illegal activities, "
                   "harmful actions, or criminal advice. If yes, flag as unsafe. "
                   "Respond with 'SAFE' or 'UNSAFE: reason'. Be conservative."),
        ("human", f"Query: {query}")
    ])
    try:
        response = llm.invoke(prompt.format_messages())
        content = response.content.strip().upper()
        if content.startswith("UNSAFE"):
            reason = content.split(":", 1)[1].strip() if ":" in content else "Potential harmful intent"
            return False, reason
        return True, "Safe"
    except Exception:
        # Fallback to keyword if LLM fails
        return not is_harmful_query(query), "Fallback check"