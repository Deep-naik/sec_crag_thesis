# 07_evaluate_ragas.py
import os
import json
import time
import pandas as pd
from importlib import import_module

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Import your compiled Phase 3 CRAG graph
crag_module = import_module("crag_agent")
crag_app = crag_module.crag_app

# Set your Groq API key (or ensure it's exported in your shell)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# 1. Initialize Embeddings, Vector Store & Rate-Limit-Safe Generator LLM
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_db = Chroma(
    persist_directory="./chroma_baseline_db",
    embedding_function=embeddings
)

generator_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=2048,
    max_retries=6
)

# 2. Vanilla Baseline RAG Function
def run_vanilla_rag(question: str):
    start_time = time.time()
    docs = vector_db.similarity_search(question, k=5)
    contexts = [d.page_content for d in docs]
    context_str = "\n\n---\n\n".join(contexts)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a financial analyst. Answer the question using the provided SEC 10-K excerpts."),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

    ans = (prompt | generator_llm).invoke({
        "question": question,
        "context": context_str
    }).content

    latency = round(time.time() - start_time, 3)
    return ans, contexts, latency

# 3. CRAG Agent Runner Function
def run_crag_agent(question: str):
    start_time = time.time()
    result = crag_app.invoke({
        "original_query": question,
        "current_query": question,
        "target_ticker": None,
        "retrieved_docs": [],
        "filtered_docs": [],
        "critique_logs": [],
        "loop_count": 0,
        "final_answer": ""
    })
    latency = round(time.time() - start_time, 3)
    contexts = [d.page_content for d in result.get("filtered_docs", [])]
    # Fallback to retrieved_docs if all chunks were rejected on a false-premise query
    if not contexts:
        contexts = [d.page_content for d in result.get("retrieved_docs", [])]
    return result["final_answer"], contexts, latency, result.get("loop_count", 0)

# 4. Main Generation & Checkpointing Pipeline
def main():
    benchmark_file = "sec_10k_benchmark_50.json"
    checkpoint_file = "eval_checkpoint.jsonl"

    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    completed_records = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed_records[rec["id"]] = rec
        print(f"Resuming from checkpoint: {len(completed_records)}/{len(benchmark_data)} queries already completed.\n")
    else:
        print(f"Running {len(benchmark_data)}-query comparative benchmark...\n")

    for item in benchmark_data:
        qid = item["id"]
        if qid in completed_records:
            continue

        q = item["question"]
        print(f"\n==================== {qid} [{item.get('category', '')}] ====================")

        v_ans, v_ctx, v_lat = run_vanilla_rag(q)
        time.sleep(8)  # Allow TPM bucket to replenish before CRAG multi-turn loop
        c_ans, c_ctx, c_lat, loops = run_crag_agent(q)

        record = {
            "id": qid,
            "ticker": item.get("ticker", ""),
            "category": item.get("category", ""),
            "question": q,
            "ground_truth": item["ground_truth"],
            "vanilla_answer": v_ans,
            "vanilla_contexts": v_ctx,
            "vanilla_latency": v_lat,
            "crag_answer": c_ans,
            "crag_contexts": c_ctx,
            "crag_latency": c_lat,
            "crag_loops": loops
        }

        completed_records[qid] = record
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(f"Completed {qid} | Vanilla: {v_lat}s | CRAG: {c_lat}s (Loops: {loops})")
        time.sleep(15)  # Pacing to stay under Groq 8,000 TPM limit

    # Order records cleanly Q01 -> Q50 and export CSV
    ordered_results = [completed_records[item["id"]] for item in benchmark_data]
    df_raw = pd.DataFrame(ordered_results)
    df_raw.to_csv("raw_benchmark_outputs.csv", index=False)
    print("\nAll 50 queries generated! Saved raw responses to raw_benchmark_outputs.csv")
    print("Next step: Run `python 08_score_saved_results.py` to compute RAGAS metrics.")

if __name__ == "__main__":
    main()