# 08_score_saved_results.py
import os
import ast
import json
import time
import warnings
import pandas as pd
from datasets import Dataset

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add one or more Groq API keys from separate organizations if rotating on TPD limits
GROQ_API_KEYS = [
    os.environ.get("GROQ_API_KEY", "your_groq_api_key_1"),
]

EVAL_MODEL = "openai/gpt-oss-20b"


def build_evaluator(api_key: str, embeddings):
    llm = ChatGroq(
        api_key=api_key,
        model=EVAL_MODEL,
        temperature=0,
        max_tokens=4096,
        reasoning_effort="low",  # Prevents hidden thinking tokens from exhausting max_tokens
        max_retries=2,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def parse_contexts(val):
    if pd.isna(val) or not str(val).strip():
        return [""]
    parsed = ast.literal_eval(str(val))
    # Keep top 4 chunks at max 900 chars each so total request stays well under Groq's 8,000 TPM limit
    trimmed = [str(c)[:900] for c in parsed[:4]]
    return trimmed if trimmed else [""]


def score_dataset_resumable(
    df: pd.DataFrame,
    answer_col: str,
    context_col: str,
    checkpoint_path: str,
    output_csv: str,
    embeddings,
):
    completed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed[rec["id"]] = rec
        print(f"Resuming {checkpoint_path}: {len(completed)}/{len(df)} rows already scored.")

    metrics = [
        Faithfulness(),
        AnswerRelevancy(strictness=1),
        ContextPrecision(),
        ContextRecall(),
    ]
    run_cfg = RunConfig(max_workers=1, timeout=180, max_wait=60, max_retries=2)

    key_idx = 0
    evaluator_llm, evaluator_embeddings = build_evaluator(GROQ_API_KEYS[key_idx], embeddings)

    for _, row in df.iterrows():
        qid = row["id"]
        if qid in completed:
            continue

        single_ds = Dataset.from_dict({
            "question": [row["question"]],
            "answer": [row[answer_col][:1500]],
            "contexts": [row[context_col]],
            "ground_truth": [row["ground_truth"][:1000]],
        })

        while True:
            try:
                res_df = evaluate(
                    dataset=single_ds,
                    metrics=metrics,
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                    run_config=run_cfg,
                    show_progress=False,
                ).to_pandas()

                scores = {
                    "id": qid,
                    "faithfulness": float(res_df["faithfulness"].iloc[0]) if pd.notna(res_df["faithfulness"].iloc[0]) else None,
                    "answer_relevancy": float(res_df["answer_relevancy"].iloc[0]) if pd.notna(res_df["answer_relevancy"].iloc[0]) else None,
                    "context_precision": float(res_df["context_precision"].iloc[0]) if pd.notna(res_df["context_precision"].iloc[0]) else None,
                    "context_recall": float(res_df["context_recall"].iloc[0]) if pd.notna(res_df["context_recall"].iloc[0]) else None,
                }

                completed[qid] = scores
                with open(checkpoint_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(scores) + "\n")

                print(
                    f"[{len(completed)}/{len(df)}] {qid} -> "
                    f"Faith: {scores['faithfulness']}, Rel: {scores['answer_relevancy']}, "
                    f"Prec: {scores['context_precision']}, Rec: {scores['context_recall']}"
                )
                time.sleep(8)  # Let 8,000 TPM rolling window replenish between rows
                break

            except Exception as e:
                err_msg = str(e)
                if "tokens per day" in err_msg.lower() or "tpd" in err_msg.lower():
                    key_idx += 1
                    if key_idx < len(GROQ_API_KEYS):
                        print(f"\n[KEY ROTATION] Daily token limit hit. Switching to API key #{key_idx + 1}...")
                        evaluator_llm, evaluator_embeddings = build_evaluator(GROQ_API_KEYS[key_idx], embeddings)
                        continue
                    else:
                        print(f"\n[STOPPED] Daily TPD quota exhausted at {len(completed)}/{len(df)} rows.")
                        print("Progress is saved! Add a key from another Groq org or re-run after reset.")
                        raise SystemExit(1)
                elif "rate_limit" in err_msg.lower() or "429" in err_msg or "413" in err_msg:
                    print("  [TPM Pause] Sleeping 30s for 8,000 TPM rolling window...")
                    time.sleep(30)
                else:
                    print(f"  [WARN] Error on {qid}: {err_msg[:140]} — retrying in 10s...")
                    time.sleep(10)

    ordered = [completed[qid] for qid in df["id"] if qid in completed]
    final_df = pd.DataFrame(ordered)
    final_df.to_csv(output_csv, index=False)
    return final_df


def main():
    df_raw = pd.read_csv("raw_benchmark_outputs.csv")

    for col in ["question", "ground_truth", "vanilla_answer", "crag_answer", "category", "ticker"]:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].fillna("").astype(str)

    df_raw["vanilla_contexts"] = df_raw["vanilla_contexts"].apply(parse_contexts)
    df_raw["crag_contexts"] = df_raw["crag_contexts"].apply(parse_contexts)

    answerable_df = df_raw[df_raw["category"] != "false_premise"].copy()
    print(f"Loaded {len(df_raw)} total rows ({len(answerable_df)} answerable, {len(df_raw) - len(answerable_df)} false-premise).")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    print("\n--- Step 1: Scoring Vanilla RAG (Resumable) ---")
    vanilla_scores = score_dataset_resumable(
        df=answerable_df,
        answer_col="vanilla_answer",
        context_col="vanilla_contexts",
        checkpoint_path="ragas_vanilla_checkpoint.jsonl",
        output_csv="ragas_vanilla_results.csv",
        embeddings=embeddings,
    )

    print("\n--- Step 2: Scoring CRAG Agent (Resumable) ---")
    crag_scores = score_dataset_resumable(
        df=answerable_df,
        answer_col="crag_answer",
        context_col="crag_contexts",
        checkpoint_path="ragas_crag_checkpoint.jsonl",
        output_csv="ragas_crag_results.csv",
        embeddings=embeddings,
    )

    # Compute False-Premise Rejection Rate (Q46-Q50) & Final Thesis Summary Table
    traps_df = df_raw[df_raw["category"] == "false_premise"]
    v_abstentions = traps_df["vanilla_answer"].str.contains(
        "INSUFFICIENT EVIDENCE|not disclose|not mention|no mention|does not contain",
        case=False, na=False
    ).sum()
    c_abstentions = traps_df["crag_answer"].str.contains(
        "INSUFFICIENT EVIDENCE", case=False, na=False
    ).sum()

    summary = pd.DataFrame({
        "Metric": [
            "Faithfulness",
            "Answer Relevancy",
            "Context Precision",
            "Context Recall",
            "False-Premise Rejection Rate (%)",
            "Avg Latency (s)"
        ],
        "Vanilla RAG": [
            round(vanilla_scores["faithfulness"].mean(), 4),
            round(vanilla_scores["answer_relevancy"].mean(), 4),
            round(vanilla_scores["context_precision"].mean(), 4),
            round(vanilla_scores["context_recall"].mean(), 4),
            round((v_abstentions / len(traps_df)) * 100, 1) if len(traps_df) else 0,
            round(df_raw["vanilla_latency"].mean(), 2)
        ],
        "Corrective RAG (Ours)": [
            round(crag_scores["faithfulness"].mean(), 4),
            round(crag_scores["answer_relevancy"].mean(), 4),
            round(crag_scores["context_precision"].mean(), 4),
            round(crag_scores["context_recall"].mean(), 4),
            round((c_abstentions / len(traps_df)) * 100, 1) if len(traps_df) else 0,
            round(df_raw["crag_latency"].mean(), 2)
        ]
    })

    print("\n" + "=" * 65)
    print("FINAL THESIS BENCHMARK RESULTS (VANILLA RAG vs. CRAG)")
    print("=" * 65)
    print(summary.to_string(index=False))
    summary.to_csv("final_thesis_summary.csv", index=False)


if __name__ == "__main__":
    main()