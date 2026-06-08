from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from app.config import settings
from app.utils.safety import is_harmful_query, llm_safety_check
from app.tools.web_search import web_search
from app.tools.youtube_transcript import get_youtube_transcript
from app.tools.case_law_search import search_indian_case_law
from app.tools.legal_research import legal_research
import concurrent.futures

tools = [web_search, get_youtube_transcript, search_indian_case_law, legal_research]

class AgentState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    is_safe: bool
    safety_reason: str
    response: str
    research_data: str

def get_llm():
    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.llm_model,
        temperature=0.2,
        max_tokens=600
    )

def input_safety_node(state: AgentState) -> AgentState:
    query = state["query"]
    if is_harmful_query(query):
        state["is_safe"] = False
        state["safety_reason"] = "Harmful intent detected"
        return state
    is_safe, reason = llm_safety_check(query)
    state["is_safe"] = is_safe
    state["safety_reason"] = reason
    return state

def research_node(state: AgentState) -> AgentState:
    query = state["query"]
    
    def run_tool(tool):
        try:
            return tool.invoke(query)
        except:
            return ""
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_tool, tools))
    
    combined_research = "\n\n".join([r for r in results if r])
    state["research_data"] = combined_research[:4000]
    return state

def output_safety_node(state: AgentState) -> AgentState:
    if not state.get("response"):
        return state
    if any(word in state["response"].lower() for word in ["illegal", "how to commit", "bypass law"]):
        state["response"] = "I cannot provide this information as it may violate legal guidelines."
    return state

def generate_response(state: AgentState) -> AgentState:
    if not state["is_safe"]:
        state["response"] = "I cannot assist with this query due to safety concerns."
        return state
    
    llm = get_llm()
    
    messages = []
    for msg in state.get("chat_history", []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    messages.append(HumanMessage(content=f"Query: {state['query']}\n\nResearch: {state.get('research_data', '')[:3000]}"))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional Indian lawyer.
        Respond in a clear, concise, and structured format using these sections:
        
        **Key Points:**
        - Brief summary
        
        **Relevant Legal Provisions:**
        - Mention sections/case laws
        
        **Analysis & Advice:**
        - Short professional guidance
        
        **Disclaimer:** 
        - Always include this at the end."""),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"messages": messages})
    state["response"] = response
    return state

def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("input_safety", input_safety_node)
    workflow.add_node("research", research_node)
    workflow.add_node("generate", generate_response)
    workflow.add_node("output_safety", output_safety_node)
    
    workflow.set_entry_point("input_safety")
    
    workflow.add_conditional_edges(
        "input_safety",
        lambda state: "research" if state["is_safe"] else END,
        {"research": "research", END: END}
    )
    
    workflow.add_edge("research", "generate")
    workflow.add_edge("generate", "output_safety")
    workflow.add_edge("output_safety", END)
    
    return workflow.compile()

def process_legal_query(query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
    if chat_history is None:
        chat_history = []
    
    agent = create_agent_graph()
    initial_state: AgentState = {
        "query": query,
        "chat_history": chat_history,
        "is_safe": True,
        "safety_reason": "",
        "response": "",
        "research_data": ""
    }
    
    result = agent.invoke(initial_state)
    
    new_history = chat_history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": result["response"]}
    ]
    
    return {
        "response": result["response"],
        "safety": {"is_safe": result["is_safe"], "reason": result["safety_reason"]},
        "chat_history": new_history
    }