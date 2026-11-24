# 06_generate_benchmark.py
import os
import json
import time
import random
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Set your active Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_db = Chroma(persist_directory="./chroma_baseline_db", embedding_function=embeddings)

# Use active Groq production model (llama-3.1-8b-instant was deprecated)
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

class BenchmarkItem(BaseModel):
    question: str = Field(description="A challenging, specific financial question answerable strictly from the excerpt.")
    ground_truth: str = Field(description="The exact, factual reference answer derived from the excerpt.")
    category: str = Field(description="One of: 'entity_specific', 'quantitative', 'multi_hop'")

def build_50_query_benchmark(output_path: str = "sec_10k_benchmark_50.json"):
    raw = vector_db.get(include=["documents", "metadatas"])
    records = list(zip(raw["documents"], raw["metadatas"]))
    random.seed(42)
    random.shuffle(records)

    structured_gen = llm.with_structured_output(BenchmarkItem, method="json_mode")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are creating an academic benchmark dataset from SEC Form 10-K filings. "
         "Given a filing excerpt and company ticker, generate ONE specific question that explicitly names "
         "the company (Apple, Microsoft, or NVIDIA) and asks for a concrete risk, operational dependency, or financial figure. "
         "Provide the exact ground-truth answer supported by the text. "
         "Respond ONLY in valid JSON with keys: 'question' (string), 'ground_truth' (string), and 'category' ('entity_specific', 'quantitative', or 'multi_hop')."),
        ("human", "Ticker: {ticker}\nExcerpt:\n{text}")
    ])
    chain = prompt | structured_gen

    benchmark = []
    ticker_counts = {"AAPL": 0, "MSFT": 0, "NVDA": 0}
    target_per_ticker = 15  # 15 * 3 = 45 grounded questions

    for text, meta in records:
        if len(benchmark) >= 45:
            break
        if len(text) < 500:
            continue
        ticker = meta.get("ticker", "UNKNOWN")
        if ticker not in ticker_counts or ticker_counts[ticker] >= target_per_ticker:
            continue

        try:
            item: BenchmarkItem = chain.invoke({"ticker": ticker, "text": text[:2000]})
            ticker_counts[ticker] += 1
            benchmark.append({
                "id": f"Q{len(benchmark)+1:02d}",
                "ticker": ticker,
                "category": item.category,
                "question": item.question,
                "ground_truth": item.ground_truth,
                "reference_context": text
            })
            print(f"Generated {len(benchmark)}/45 [{ticker} {ticker_counts[ticker]}/{target_per_ticker}]: {item.question[:65]}...")
            time.sleep(1.5)  # Respect Groq free-tier rate limits
        except Exception as e:
            print(f"  [SKIP] Generation error on {ticker} chunk: {e}")
            time.sleep(2)
            continue

    # Append 5 adversarial False-Premise / Unanswerable questions (Hallucination Traps)
    false_premises = [
        {
            "id": "Q46", "ticker": "AAPL", "category": "false_premise",
            "question": "What regulatory fine did Apple disclose in its 10-K for the forced divestiture of its App Store payment division?",
            "ground_truth": "INSUFFICIENT EVIDENCE: The SEC 10-K filings do not report any forced divestiture of the App Store payment division.",
            "reference_context": ""
        },
        {
            "id": "Q47", "ticker": "NVDA", "category": "false_premise",
            "question": "How much revenue did NVIDIA generate from commercial nuclear fusion reactor licensing in fiscal year 2024?",
            "ground_truth": "INSUFFICIENT EVIDENCE: The SEC 10-K filings contain no disclosures regarding commercial nuclear fusion reactor licensing revenue.",
            "reference_context": ""
        },
        {
            "id": "Q48", "ticker": "MSFT", "category": "false_premise",
            "question": "What was the exact acquisition price Microsoft paid to acquire Oracle Corporation according to its latest 10-K?",
            "ground_truth": "INSUFFICIENT EVIDENCE: Microsoft did not acquire Oracle Corporation in the SEC 10-K filings.",
            "reference_context": ""
        },
        {
            "id": "Q49", "ticker": "AAPL", "category": "false_premise",
            "question": "Which European central bank bailed out Apple's Irish subsidiary according to Item 7 of its 10-K?",
            "ground_truth": "INSUFFICIENT EVIDENCE: No European central bank bailout of Apple's Irish subsidiary is disclosed in the 10-K filings.",
            "reference_context": ""
        },
        {
            "id": "Q50", "ticker": "NVDA", "category": "false_premise",
            "question": "What percentage of NVIDIA's GPU wafers were manufactured in Brazil according to its 10-K supply chain disclosures?",
            "ground_truth": "INSUFFICIENT EVIDENCE: NVIDIA's 10-K filings do not report GPU wafer manufacturing in Brazil.",
            "reference_context": ""
        }
    ]
    benchmark.extend(false_premises)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\nSaved {len(benchmark)}-question benchmark to {output_path} (Counts: {ticker_counts} + 5 false_premise)")

if __name__ == "__main__":
    build_50_query_benchmark()