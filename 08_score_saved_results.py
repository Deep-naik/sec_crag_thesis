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

# Put your active Groq API key(s) here (from separate Groq orgs if rotating)
GROQ_API_KEYS = [
    os.environ.get("GROQ_API_KEY", "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"),
]

EVAL_MODEL = "openai/gpt-oss-20b"
METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def build_evaluator(api_key: str, embeddings):
    llm = ChatGroq(
        api_key=api_key,
        model=EVAL_MODEL,
        temperature=0,
        max_tokens=4096,
        reasoning_effort="low",
        max_retries=2,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def parse_contexts(val):
    if pd.isna(val) or not str(val).strip():
        return [""]
    parsed = ast.literal_eval(str(val))
    trimmed = [str(c)[:800] for c in parsed[:4]]
    return trimmed if trimmed else [""]


def is_valid_record(rec: dict) -> bool:
    # Require at least 3 of 4 metrics to be non-None so a single metric parse hiccup doesn't discard the row
    valid_count = sum(1 for k in METRIC_KEYS if rec.get(k) is not None)
    return valid_count >= 3


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
                    if is_valid_record(rec):
                        completed[rec["id"]] = rec
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            for rec in completed.values():
                f.write(json.dumps(rec) + "\n")
        print(f"Resuming {checkpoint_path}: {len(completed)}/{len(df)} valid rows already scored.")

    metrics = [
        Faithfulness(),
        AnswerRelevancy(strictness=1),
        ContextPrecision(),
        ContextRecall(),
    ]
    run_cfg = RunConfig(max_workers=1, timeout=180, max_wait=45, max_retries=2)

    key_idx = 0
    evaluator_llm, evaluator_embeddings = build_evaluator(GROQ_API_KEYS[key_idx], embeddings)
    global_tpd_strikes = 0

    for _, row in df.iterrows():
        qid = row["id"]
        if qid in completed:
            continue

        single_ds = Dataset.from_dict({
            "question": [row["question"]],
            "answer": [row[answer_col][:1200]],
            "contexts": [row[context_col]],
            "ground_truth": [row["ground_truth"][:900]],
        })

        row_attempts = 0
        while row_attempts < 3:
            row_attempts += 1
            try:
                res_df = evaluate(
                    dataset=single_ds,
                    metrics=metrics,
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                    run_config=run_cfg,
                    show_progress=True,  # Shows exact Groq/RAGAS error if any metric fails
                ).to_pandas()

                scores = {
                    "id": qid,
                    "faithfulness": float(res_df["faithfulness"].iloc[0]) if pd.notna(res_df["faithfulness"].iloc[0]) else None,
                    "answer_relevancy": float(res_df["answer_relevancy"].iloc[0]) if pd.notna(res_df["answer_relevancy"].iloc[0]) else None,
                    "context_precision": float(res_df["context_precision"].iloc[0]) if pd.notna(res_df["context_precision"].iloc[0]) else None,
                    "context_recall": float(res_df["context_recall"].iloc[0]) if pd.notna(res_df["context_recall"].iloc[0]) else None,
                }

                # If all 4 are None, it's a true 429 rate-limit/TPD block
                if all(scores[k] is None for k in METRIC_KEYS):
                    raise RuntimeError("All 4 metrics returned None (429 rate limit or TPD cap).")

                global_tpd_strikes = 0
                completed[qid] = scores
                with open(checkpoint_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(scores) + "\n")

                print(
                    f"[{len(completed)}/{len(df)}] {qid} -> "
                    f"Faith: {scores['faithfulness']}, Rel: {scores['answer_relevancy']}, "
                    f"Prec: {scores['context_precision']}, Rec: {scores['context_recall']}"
                )
                time.sleep(40)  # Keep rolling usage below 8,000 TPM
                break

            except Exception as e:
                print(f"  [ATTEMPT {row_attempts}/3 on {qid}] {str(e)[:120]} — waiting 60s...")
                time.sleep(60)
        else:
            # Row failed 3 times (likely a row-specific JSON parse issue on Q03); skip row & continue!
            global_tpd_strikes += 1
            print(f"  [SKIP ROW] {qid} failed 3 attempts; saving placeholder and moving to next query...")
            fallback = {"id": qid, "faithfulness": None, "answer_relevancy": None, "context_precision": None, "context_recall": None}
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(fallback) + "\n")

            # Only stop if 3 DIFFERENT consecutive rows completely fail (true daily TPD exhaustion)
            if global_tpd_strikes >= 3:
                key_idx += 1
                if key_idx < len(GROQ_API_KEYS):
                    print(f"\n[KEY ROTATION] 3 consecutive rows failed. Switching to API key #{key_idx + 1}...")
                    evaluator_llm, evaluator_embeddings = build_evaluator(GROQ_API_KEYS[key_idx], embeddings)
                    global_tpd_strikes = 0
                else:
                    print(f"\n[PAUSED] 3 consecutive rows failed (TPD quota exhausted) at {len(completed)}/{len(df)} rows.")
                    print("Progress saved. Add a fresh key from another Groq org or run again after quota resets.")
                    raise SystemExit(1)

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