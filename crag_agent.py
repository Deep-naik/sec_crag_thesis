# 04_crag_agent.py
import os
from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from hybrid_reranker import HybridDynamicReranker
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# 1. Initialize Vector Store & LLM
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_db = Chroma(
    persist_directory="./chroma_baseline_db", 
    embedding_function=embeddings
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# 2. Define Pydantic Schemas for Deterministic Agent Decisions
class ChunkCritique(BaseModel):
    is_relevant: bool = Field(
        description="True ONLY if the chunk directly addresses the target company, fiscal year, and specific financial metric/risk asked."
    )
    groundedness_score: float = Field(
        description="Relevance and specificity score between 0.0 and 1.0."
    )
    critique_reason: str = Field(
        description="Brief explanation identifying any entity mismatch, year drift, or boilerplate noise."
    )

class QueryRewrite(BaseModel):
    rewritten_query: str = Field(
        description="An expanded, keyword-rich financial search query optimized for SEC 10-K retrieval."
    )
    target_ticker: Optional[str] = Field(
        default=None,
        description="Extracted stock ticker if a specific company is mentioned: 'AAPL', 'MSFT', 'NVDA', or None."
    )

# 3. Define the Shared Graph State
class CRAGState(TypedDict):
    original_query: str
    current_query: str
    target_ticker: Optional[str]
    retrieved_docs: List[Document]
    filtered_docs: List[Document]
    critique_logs: List[str]
    loop_count: int
    final_answer: str

# 4. Define Graph Nodes
reranker_engine = HybridDynamicReranker(persist_directory="./chroma_baseline_db")

def retrieve_node(state: CRAGState) -> dict:
    """Two-Stage Hybrid Retrieval + Dynamic RRF + Cross-Encoder Reranking."""
    query = state["current_query"]
    ticker = state.get("target_ticker")
    loop = state.get("loop_count", 0)

    print(f"\n[NODE: HYBRID RETRIEVE | Loop {loop}] Query: '{query}' | Filter: {ticker}")

    docs = reranker_engine.retrieve_and_rerank(
        query=query,
        target_ticker=ticker,
        stage1_k=25,
        final_top_k=5,
    )
    return {"retrieved_docs": docs}

def critic_node(state: CRAGState) -> dict:
    """Adversarial Critic Agent: inspects each chunk for entity drift, year drift, and boilerplate noise."""
    print("[NODE: ADVERSARIAL CRITIC] Evaluating retrieved chunks...")
    
    structured_critic = llm.with_structured_output(ChunkCritique, method="json_mode")
    
    critic_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a strict financial audit Critic for SEC 10-K filings. Do NOT answer the user's question. "
         "Your ONLY job is to evaluate whether the Document Text supports answering the User Question. "
         "Reject a chunk if: (1) It belongs to the wrong company, (2) It discusses the wrong fiscal year, "
         "or (3) It is generic legal boilerplate that lacks the specific facts requested. "
         "You MUST respond ONLY with a JSON object containing these exact three keys: "
         "'is_relevant' (boolean), 'groundedness_score' (float between 0.0 and 1.0), and 'critique_reason' (string)."),
        ("human", 
         "User Question: {question}\n\n"
         "Document Metadata: {metadata}\n"
         "Document Text: {context}")
    ])
    
    critic_chain = critic_prompt | structured_critic
    
    valid_docs = []
    logs = state.get("critique_logs", [])
    
    for idx, doc in enumerate(state["retrieved_docs"], 1):
        try:
            verdict: ChunkCritique = critic_chain.invoke({
                "question": state["original_query"],
                "metadata": str(doc.metadata),
                "context": doc.page_content
            })
            is_rel = verdict.is_relevant
            score = verdict.groundedness_score
            reason = verdict.critique_reason
        except Exception as e:
            print(f"  [CRITIC PARSE FALLBACK] Chunk {idx} schema mismatch ({str(e)[:80]}); marking REJECT.")
            is_rel = False
            score = 0.0
            reason = "Output parser schema mismatch fallback."
        
        status = "PASS" if (is_rel and score >= 0.70) else "REJECT"
        log_entry = f"Chunk {idx} [{doc.metadata.get('ticker')}]: {status} (Score: {score:.2f}) - {reason}"
        print("  " + log_entry)
        logs.append(log_entry)
        
        if status == "PASS":
            valid_docs.append(doc)
            
    return {"filtered_docs": valid_docs, "critique_logs": logs}


def rewrite_node(state: CRAGState) -> dict:
    """Corrective Sub-Routine: reformulates query and extracts explicit metadata filters."""
    print("[NODE: QUERY REWRITER] Insufficient valid context. Formulating corrective query...")
    
    structured_rewriter = llm.with_structured_output(QueryRewrite, method="json_mode")
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a financial information retrieval specialist. The previous vector search returned "
         "irrelevant chunks or wrong-company boilerplate. Rewrite the query using precise SEC 10-K terminology "
         "(e.g., Item 1A Risk Factors, Item 7 MD&A, net sales, operating income) and extract the exact stock ticker "
         "('AAPL', 'MSFT', or 'NVDA') if applicable. "
         "Respond ONLY in valid JSON with keys: 'rewritten_query' (string) and 'target_ticker' ('AAPL', 'MSFT', 'NVDA', or null)."),
        ("human", 
         "Original Query: {original_query}\n"
         "Previous Failed Query: {current_query}\n"
         "Critic Feedback: {feedback}")
    ])
    
    rewriter_chain = rewrite_prompt | structured_rewriter
    
    try:
        result: QueryRewrite = rewriter_chain.invoke({
            "original_query": state["original_query"],
            "current_query": state["current_query"],
            "feedback": "\n".join(state["critique_logs"][-5:])
        })
        new_query = result.rewritten_query
        new_ticker = result.target_ticker or state.get("target_ticker")
    except Exception as e:
        print(f"  [REWRITER FALLBACK] Structured parse failed ({e}); keeping query and incrementing loop.")
        new_query = state["current_query"]
        new_ticker = state.get("target_ticker")
    
    return {
        "current_query": new_query,
        "target_ticker": new_ticker,
        "loop_count": state.get("loop_count", 0) + 1
    }
    """Corrective Sub-Routine: reformulates query and extracts explicit metadata filters."""
    print("[NODE: QUERY REWRITER] Insufficient valid context. Formulating corrective query...")
    
    structured_rewriter = llm.with_structured_output(QueryRewrite)
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a financial information retrieval specialist. The previous vector search returned "
         "irrelevant chunks or wrong-company boilerplate. Rewrite the query using precise SEC 10-K terminology "
         "(e.g., Item 1A Risk Factors, Item 7 MD&A, net sales, operating income) and extract the exact stock ticker "
         "('AAPL', 'MSFT', or 'NVDA') if applicable."),
        ("human", 
         "Original Query: {original_query}\n"
         "Previous Failed Query: {current_query}\n"
         "Critic Feedback: {feedback}")
    ])
    
    rewriter_chain = rewrite_prompt | structured_rewriter
    result: QueryRewrite = rewriter_chain.invoke({
        "original_query": state["original_query"],
        "current_query": state["current_query"],
        "feedback": "\n".join(state["critique_logs"][-5:])
    })
    
    return {
        "current_query": result.rewritten_query,
        "target_ticker": result.target_ticker or state.get("target_ticker"),
        "loop_count": state.get("loop_count", 0) + 1
    }

def generate_node(state: CRAGState) -> dict:
    """Synthesizes final answer strictly from Critic-approved chunks."""
    print("[NODE: GENERATOR] Synthesizing final grounded response...")
    
    valid_docs = state["filtered_docs"]
    if not valid_docs:
        return {
            "final_answer": (
                "INSUFFICIENT EVIDENCE: After self-correction loops, the Adversarial Critic found no "
                "grounded evidence in the SEC 10-K corpus to answer this query without hallucination."
            )
        }
        
    context_text = "\n\n---\n\n".join(
        [f"\n{d.page_content}" for d in valid_docs]
    )
    
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a senior financial analyst. Answer the user's question using ONLY the verified SEC 10-K "
         "excerpts below. Cite the company ticker for every claim. Do not extrapolate."),
        ("human", "Question: {question}\n\nVerified Context:\n{context}")
    ])
    
    response = (gen_prompt | llm).invoke({
        "question": state["original_query"],
        "context": context_text
    })
    
    return {"final_answer": response.content}

# 5. Conditional Routing Logic
def route_after_critic(state: CRAGState) -> str:
    """Decides whether to proceed to generation or trigger the corrective rewrite loop."""
    if len(state["filtered_docs"]) >= 2:
        return "generate"
    if state.get("loop_count", 0) >= 2:
        print("[ROUTER] Max correction loops (2) reached. Proceeding with available evidence.")
        return "generate"
    return "rewrite"

# 6. Compile the StateGraph
workflow = StateGraph(CRAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("critic", critic_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "critic")
workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

crag_app = workflow.compile()

# 7. Test Against the Baseline Failure Queries
if __name__ == "__main__":
    # Test Query #1 (Cross-Entity Trap) and Query #5 (False-Premise Trap)
    test_queries = [
        "What specific export control restrictions and licensing requirements did NVIDIA report regarding high-performance integrated circuits shipped to China?",
        "What penalties or regulatory fines did Apple disclose in its 10-K regarding the forced divestiture of its App Store payment processing division?"
    ]
    
    for q in test_queries:
        print("=" * 80)
        print(f"TESTING QUERY: {q}")
        result = crag_app.invoke({
            "original_query": q,
            "current_query": q,
            "target_ticker": None,
            "retrieved_docs": [],
            "filtered_docs": [],
            "critique_logs": [],
            "loop_count": 0,
            "final_answer": ""
        })
        print("\nFINAL OUTPUT:\n", result["final_answer"])