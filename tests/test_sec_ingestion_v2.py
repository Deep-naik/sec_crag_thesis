import importlib.util
from pathlib import Path
from bs4 import BeautifulSoup

spec=importlib.util.spec_from_file_location("sec_ingest",Path("09_ingest_sec_v2.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

SAMPLE="""<?xml version="1.0"?>
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
<body>
<table><tr><td><div>Item 7. Management Discussion and Analysis</div></td></tr>
<tr><td><div>Item 8. Financial Statements and Supplementary Data</div></td></tr></table>
<ix:nonFraction name="us-gaap:Revenue" contextRef="FY" unitRef="USD" scale="6" decimals="-3">123.5</ix:nonFraction>
<div>Item 7. Management Discussion and Analysis</div>
<div>Actual Item 7 narrative with revenue, operating costs, and a sufficiently long discussion for extraction.</div>
<div>Item 8. Financial Statements and Supplementary Data</div>
<div>Actual Item 8 narrative with financial statements, balance sheet, and cash flow information.</div>
<div>Item 10. Directors and Corporate Governance</div>
<div>Later section content should not be included in Item 8.</div>
</body></html>"""

def test_inline_fact_preserves_numeric_metadata():
    soup=BeautifulSoup(SAMPLE,"xml")
    filing={"ticker":"TST","issuer":"Test issuer","cik":"0000000001","accession_number":"0000000001-25-000001","filing_date":"2025-02-01","fiscal_period_end":"2024-12-31"}
    fact=mod.extract_inline_facts(soup,filing,"sample.html")[0]
    assert fact["concept"]=="us-gaap:Revenue"
    assert fact["value"]==123500000.0
    assert fact["unit"]=="USD"
    assert fact["scale"]=="6" and fact["decimals"]=="-3"

def test_item_sections_use_body_headings_and_exclude_toc():
    sections=mod.extract_item_sections(BeautifulSoup(SAMPLE,"xml"))
    assert "7" in sections and "8" in sections
    assert "Actual Item 7 narrative" in sections["7"]
    assert "Actual Item 8 narrative" in sections["8"]
    assert "Later section content" not in sections["8"]
    assert "table" not in sections["7"].lower()

def test_filing_metadata_reads_dei_and_sec_submission_header(tmp_path):
    filing_dir=tmp_path/"AAPL"/"10-K"/"0000320193-26-000026"
    filing_dir.mkdir(parents=True)
    html=filing_dir/"primary-document.html"
    html.write_text(
        '<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>'
        '<ix:nonNumeric name="dei:EntityCentralIndexKey">0000320193</ix:nonNumeric>'
        '<ix:nonNumeric name="dei:DocumentPeriodEndDate">September 26, 2026</ix:nonNumeric>'
        '</body></html>',
        encoding="utf-8"
    )
    (filing_dir/"full-submission.txt").write_text(
        "ACCESSION NUMBER: 0000320193-26-000026\n"
        "CONFORMED PERIOD OF REPORT: 20260926\n"
        "FILED AS OF DATE: 20261103\n"
        "FILER:\n COMPANY DATA:\n CENTRAL INDEX KEY: 0000320193\n"
        "</SEC-HEADER>",
        encoding="utf-8"
    )
    soup=BeautifulSoup(html.read_bytes(),"xml")
    metadata=mod.filing_metadata(html,soup)
    assert metadata["cik"]=="0000320193"
    assert metadata["filing_date"]=="2026-11-03"
    assert metadata["fiscal_period_end"]=="2026-09-26"
    assert metadata["accession_number"]=="0000320193-26-000026"