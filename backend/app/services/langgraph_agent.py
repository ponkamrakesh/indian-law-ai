from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from app.config import settings
from app.utils.lang_detect import detect_language
from app.utils.safety import is_harmful_query, llm_safety_check
from app.services.vector_store import get_vectorstore, search_similar
from app.services.cache import SimpleTTLCache

# Create cache instance
query_cache = SimpleTTLCache(ttl=3600)


class AgentState(TypedDict):
    query: str
    language: Dict[str, str]
    is_safe: bool
    safety_reason: str
    retrieved_docs: List[Dict]
    response: str
    metadata: Dict[str, Any]


def get_llm():
    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.llm_model,
        temperature=0.1,
        max_tokens=1024
    )


def detect_lang_node(state: AgentState) -> AgentState:
    lang_info = detect_language(state["query"])
    state["language"] = lang_info
    return state


def safety_node(state: AgentState) -> AgentState:
    query = state["query"]
    if is_harmful_query(query):
        state["is_safe"] = False
        state["safety_reason"] = "Harmful intent detected via keywords"
        return state
    
    is_safe, reason = llm_safety_check(query)
    state["is_safe"] = is_safe
    state["safety_reason"] = reason
    return state


def retrieve_node(state: AgentState) -> AgentState:
    query = state["query"]
    
    cached = query_cache.get(query)
    if cached:
        state["retrieved_docs"] = cached.get("docs", [])
        state["metadata"]["cached"] = True
        return state
    
    vectorstore = get_vectorstore()
    docs = search_similar(vectorstore, query)
    
    retrieved = []
    for doc in docs:
        retrieved.append({
            "content": doc.page_content,           # Fixed
            "metadata": doc.metadata
        })
    
    state["retrieved_docs"] = retrieved
    state["metadata"]["cached"] = False
    query_cache.set(query, {"docs": retrieved})
    return state


def generate_node(state: AgentState) -> AgentState:
    if not state["is_safe"]:
        state["response"] = f"I'm sorry, but I cannot assist with that query as it appears to involve harmful or illegal activities. {settings.disclaimer}"
        return state
    
    llm = get_llm()
    docs = state["retrieved_docs"]
    lang = state["language"]["name"]
    query = state["query"]
    
    context = "\n\n".join([
        f"Section from {d['metadata'].get('act', 'Law')}: {d['content']}" 
        for d in docs[:settings.top_k]                    # Fixed
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert Indian Law Intelligence Assistant (IndianLawAI).
        Provide accurate, helpful information based ONLY on the provided legal context.
        Respond in the detected language: {lang}.
        Always include the disclaimer at the end.
        Structure response: 
        1. Direct answer to the situation/query.
        2. Relevant sections/codes with explanations.
        3. Important notes or caveats.
        Be precise, cite sections where possible.
        If context insufficient, say so and suggest official verification.
        
        DISCLAIMER: {settings.disclaimer}"""),
        ("human", f"Query: {query}\n\nRelevant Legal Context:\n{context}\n\nProvide response in {lang}:")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({})
    state["response"] = response
    state["metadata"]["num_sources"] = len(docs)
    return state


def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("detect_lang", detect_lang_node)
    workflow.add_node("safety", safety_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("detect_lang")
    workflow.add_edge("detect_lang", "safety")
    
    # Fixed conditional routing
    def route_after_safety(state: AgentState):
        if state["is_safe"]:
            return "retrieve"
        else:
            return "end"
    
    workflow.add_conditional_edges(
        "safety",
        route_after_safety,
        {
            "retrieve": "retrieve",
            "end": END
        }
    )
    
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


def process_legal_query(query: str) -> Dict[str, Any]:
    agent = create_agent_graph()
    initial_state: AgentState = {
        "query": query,
        "language": {"code": "en", "name": "English"},
        "is_safe": True,
        "safety_reason": "",
        "retrieved_docs": [],
        "response": "",
        "metadata": {"cached": False, "num_sources": 0}
    }
    
    result = agent.invoke(initial_state)
    return {
        "response": result["response"],
        "language": result["language"],
        "sources": result["retrieved_docs"],
        "metadata": result["metadata"],
        "safety": {"is_safe": result["is_safe"], "reason": result["safety_reason"]}
    }