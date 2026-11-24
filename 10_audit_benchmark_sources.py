"""Audit benchmark reference passages against source filings and the v2 manifest."""
import json, re, csv
from pathlib import Path
from bs4 import BeautifulSoup

ROOT=Path("sec_data/sec-edgar-filings")
BENCH=Path("sec_10k_benchmark_50.json")
OUT=Path("sec_processed_v2/benchmark_source_audit.csv")

def norm(s): return re.sub(r"\s+"," ",s or "").strip().lower()

def main():
    manifest=json.loads(Path("sec_processed_v2/filing_manifest.json").read_text(encoding="utf-8"))
    sources=[]
    for row in manifest:
        p=Path(row["source_file"])
        soup=BeautifulSoup(p.read_bytes(),"lxml")
        text=norm(soup.get_text(" ",strip=True))
        sources.append((row,text))
    benchmark=json.loads(BENCH.read_text(encoding="utf-8"))
    results=[]
    for q in benchmark:
        ref=norm(q.get("reference_context",""))
        if q.get("category")=="false_premise":
            results.append({"id":q["id"],"ticker":q.get("ticker"),"status":"UNANSWERABLE_TRAP","matched_accession":"","matched_cik":"","filing_date":"","fiscal_period_end":"","reason":"No positive reference passage expected"})
            continue
        # Longest exact normalized excerpt match is conservative; absent match remains explicitly unresolved.
        matches=[]
        if ref:
            for row,text in sources:
                if ref[:min(250,len(ref))] in text:
                    matches.append(row)
        matches=[m for m in matches if m.get("ticker")==q.get("ticker")]
        if len(matches)==1:
            m=matches[0]; status="MATCHED_SINGLE_SOURCE"
            reason="Reference excerpt prefix found in exactly one filing for the benchmark ticker"
        elif len(matches)>1:
            m={}; status="AMBIGUOUS_MULTI_SOURCE"; reason="Excerpt prefix occurs in multiple filings; accession/year must be resolved"
        else:
            m={}; status="UNMATCHED_REVIEW_REQUIRED"; reason="No exact excerpt-prefix match for this ticker; do not assume reference is valid"
        results.append({"id":q["id"],"ticker":q.get("ticker"),"status":status,"matched_accession":m.get("accession_number",""),"matched_cik":m.get("cik",""),"filing_date":m.get("filing_date",""),"fiscal_period_end":m.get("fiscal_period_end",""),"reason":reason})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    print(f"Wrote {OUT}: {len(results)} benchmark rows")
    for r in results: print(r["id"],r["status"],r["matched_accession"])

if __name__=="__main__": main()
