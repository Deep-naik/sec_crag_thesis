import importlib.util
from pathlib import Path
from bs4 import BeautifulSoup

spec=importlib.util.spec_from_file_location("sec_ingest",Path("09_ingest_sec_v2.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
SAMPLE="""<html><body><xbrli:context id="FY"><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:context><xbrli:unit id="USD"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit><ix:nonFraction name="us-gaap:Revenue" contextRef="FY" unitRef="USD" scale="6" decimals="-3">123.5</ix:nonFraction><p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS This section describes revenue and costs.</p><table><tr><td>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</td></tr></table></body></html>"""

def test_inline_fact_preserves_numeric_metadata():
    soup=BeautifulSoup(SAMPLE,"lxml")
    filing={"ticker":"TST","issuer":"Test issuer","cik":"0000000001","accession_number":"0000000001-25-000001","filing_date":"2025-02-01","fiscal_period_end":"2024-12-31"}
    fact=mod.extract_inline_facts(soup,filing,"sample.html")[0]
    assert fact["concept"]=="us-gaap:Revenue"
    assert fact["value"]==123500000.0
    assert fact["unit"]=="iso4217:USD"
    assert fact["scale"]=="6" and fact["decimals"]=="-3"
    assert fact["context"]["period_start"]=="2024-01-01"
    assert fact["context"]["period_end"]=="2024-12-31"

def test_item_sections_extract_item_7_and_8():
    sections=mod.extract_item_sections(BeautifulSoup(SAMPLE,"lxml"))
    assert "7" in sections and "8" in sections
