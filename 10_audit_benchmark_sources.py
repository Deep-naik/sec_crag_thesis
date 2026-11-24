"""Audit benchmark references against original SEC filings and the v2 manifest."""

import csv
import json
import re
import warnings
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

ROOT = Path("sec_data/sec-edgar-filings")
BENCH = Path("sec_10k_benchmark_50.json")
MANIFEST = Path("sec_processed_v2/filing_manifest.json")
OUT = Path("sec_processed_v2/benchmark_source_audit.csv")

PREFIX_LENGTH = 250


def norm(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def load_sources(manifest):
    """Load and normalize filing text while retaining manifest metadata."""
    sources = []

    # SEC filings may contain XHTML/XML-like content.
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    for row in manifest:
        path = Path(row["source_file"])

        if not path.exists():
            print(f"WARNING: Source file missing: {path}")
            continue

        soup = BeautifulSoup(path.read_bytes(), "lxml")
        text = norm(soup.get_text(" ", strip=True))

        sources.append({
            **row,
            "_normalized_text": text
        })

    return sources


def audit_question(question, sources):
    ticker = question.get("ticker", "")
    reference = norm(question.get("reference_context", ""))

    base = {
        "id": question.get("id", ""),
        "ticker": ticker,
        "category": question.get("category", ""),
        "question": question.get("question", ""),
        "ground_truth": question.get("ground_truth", ""),
        "reference_chars": len(reference),
    }

    if question.get("category") == "false_premise":
        return {
            **base,
            "status": "UNANSWERABLE_TRAP",
            "match_method": "NOT_APPLICABLE",
            "match_count": 0,
            "matched_accessions": "[]",
            "matched_ciks": "[]",
            "matched_filing_dates": "[]",
            "matched_fiscal_periods": "[]",
            "matched_source_hashes": "[]",
            "reason": "False-premise question; no positive reference expected."
        }

    ticker_sources = [
        source for source in sources
        if source.get("ticker") == ticker
    ]

    # First try the complete normalized reference passage.
    full_matches = [
        source for source in ticker_sources
        if reference and reference in source["_normalized_text"]
    ]

    if full_matches:
        matches = full_matches
        method = "FULL_REFERENCE"

        if len(matches) == 1:
            status = "FULL_EXACT_UNIQUE"
            reason = "Complete normalized reference found in one filing."
        else:
            status = "FULL_EXACT_MULTIPLE"
            reason = "Complete normalized reference found in multiple filings."

    else:
        # Fall back to the first 250 normalized characters.
        prefix = reference[:PREFIX_LENGTH]

        prefix_matches = [
            source for source in ticker_sources
            if prefix and prefix in source["_normalized_text"]
        ]

        matches = prefix_matches
        method = "250_CHAR_PREFIX"

        if len(matches) == 1:
            status = "PREFIX_ONLY_UNIQUE"
            reason = "Only the reference prefix matched one filing; review required."
        elif len(matches) > 1:
            status = "PREFIX_ONLY_MULTIPLE"
            reason = "Reference prefix matched multiple filings; review required."
        else:
            status = "NO_MATCH"
            reason = "Neither full reference nor prefix matched any filing."

    return {
        **base,
        "status": status,
        "match_method": method,
        "match_count": len(matches),
        "matched_accessions": json.dumps(
            [m.get("accession_number", "") for m in matches]
        ),
        "matched_ciks": json.dumps(
            [m.get("cik", "") for m in matches]
        ),
        "matched_filing_dates": json.dumps(
            [m.get("filing_date", "") for m in matches]
        ),
        "matched_fiscal_periods": json.dumps(
            [m.get("fiscal_period_end", "") for m in matches]
        ),
        "matched_source_hashes": json.dumps(
            [m.get("source_sha256", "") for m in matches]
        ),
        "reason": reason
    }


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCH.read_text(encoding="utf-8"))

    sources = load_sources(manifest)
    results = [
        audit_question(question, sources)
        for question in benchmark
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    counts = Counter(row["status"] for row in results)

    print(f"\nSource files loaded: {len(sources)}")
    print(f"Benchmark questions audited: {len(results)}")
    print(f"Audit written to: {OUT}")

    print("\nSTATUS SUMMARY")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")

    print("\nQUESTIONS REQUIRING REVIEW")
    for row in results:
        if row["status"] in {
            "PREFIX_ONLY_UNIQUE",
            "PREFIX_ONLY_MULTIPLE",
            "NO_MATCH"
        }:
            print(
                row["id"],
                "|", row["status"],
                "|", row["matched_accessions"]
            )


if __name__ == "__main__":
    main()
