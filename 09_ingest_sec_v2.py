"""Section-aware SEC 10-K and Inline XBRL ingestion. Reads source files without modifying them."""
from __future__ import annotations
import csv, hashlib, json, re
from pathlib import Path
from typing import Any
from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_ROOT=Path("sec_data/sec-edgar-filings")
OUTPUT=Path("sec_processed_v2")
VECTOR_DIR=Path("chroma_sec_v2")
NARRATIVE_COLLECTION="sec_narrative_v2"
FACT_COLLECTION="sec_xbrl_facts_v2"
TICKERS={"AAPL","MSFT","NVDA"}
ITEMS={"1":"Business","1A":"Risk Factors","7":"Management Discussion and Analysis","8":"Financial Statements and Supplementary Data"}

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def clean(value: str) -> str:
    return re.sub(r"\s+"," ",value or "").strip()

def suffix(node, endings):
    return node.find(lambda t:isinstance(t,Tag) and t.name and any(t.name.lower().endswith(x) for x in endings))

def parse_contexts(soup):
    contexts, units={},{}
    for node in soup.find_all(lambda t:isinstance(t,Tag) and t.name and t.name.lower().endswith("context")):
        cid=node.get("id")
        if not cid: continue
        start=suffix(node,("startdate",)); end=suffix(node,("enddate",)); instant=suffix(node,("instant",))
        dims=[]
        for m in node.find_all(lambda t:isinstance(t,Tag) and t.name and (t.name.lower().endswith("explicitmember") or t.name.lower().endswith("typedmember"))):
            dims.append({"dimension":m.get("dimension"),"member":clean(m.get_text(" ",strip=True)),"kind":m.name.split(":")[-1]})
        contexts[cid]={"period_start":clean(start.get_text()) if start else None,"period_end":clean(end.get_text()) if end else None,"instant":clean(instant.get_text()) if instant else None,"dimensions":dims}
    for node in soup.find_all(lambda t:isinstance(t,Tag) and t.name and t.name.lower().endswith("unit")):
        if node.get("id"): units[node["id"]]=clean(node.get_text(" ",strip=True))
    return contexts,units

def extract_inline_facts(soup, filing, source):
    contexts,units=parse_contexts(soup); facts=[]
    tags=soup.find_all(lambda t:isinstance(t,Tag) and t.name and t.name.lower().split(":")[-1] in {"nonfraction","nonnumeric","fraction"})
    for i,node in enumerate(tags,1):
        name=node.get("name")
        if not name: continue
        cref=node.get("contextref") or node.get("contextRef")
        uref=node.get("unitref") or node.get("unitRef")
        raw=clean(node.get_text(" ",strip=True)); scale=node.get("scale"); sign=node.get("sign")
        value=raw
        if node.name.lower().split(":")[-1]=="nonfraction":
            numeric=re.sub(r"[^0-9.\-]","",raw.replace(",",""))
            try:
                value=float(numeric) if numeric else None
                if value is not None and sign=="-": value=-value
                if value is not None and scale is not None: value*=10**int(scale)
            except (ValueError,OverflowError): value=raw
        facts.append({**filing,"fact_id":f"{filing['accession_number']}:{i}","concept":name,"value":value,"raw_value":raw,"unit":units.get(uref,uref),"unit_ref":uref,"scale":scale,"decimals":node.get("decimals"),"context_ref":cref,"context":contexts.get(cref,{}),"source_file":source,"source_location":f"inline_fact_ordinal:{i}","format":node.get("format")})
    return facts

def inline_fact_by_name(soup, concept):
    """Return the first Inline XBRL fact with an exact name attribute."""
    wanted=concept.lower()
    for node in soup.find_all(lambda t:isinstance(t,Tag) and t.get("name")):
        if node.get("name","").lower()==wanted:
            return clean(node.get_text(" ",strip=True))
    return None

def iso_date(value):
    """Normalize common SEC/DEI date representations to YYYY-MM-DD."""
    value=clean(value).replace("\u00a0"," ")
    if not value:
        return None
    m=re.search(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$",value)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m=re.search(r"^([A-Za-z]+)\s+(\d{1,2})\s*,?\s*(\d{4})$",value)
    if m:
        from datetime import datetime
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}","%B %d %Y").date().isoformat()
        except ValueError:
            return value
    return value

def sec_submission_header(path):
    """Read filing metadata from the companion SEC full-submission.txt header."""
    submission=path.parent/"full-submission.txt"
    if not submission.exists():
        return {}
    text=submission.read_text(encoding="utf-8",errors="replace")
    header=text.split("</SEC-HEADER>",1)[0]
    result={}
    for key,pattern in {
        "filing_date":r"FILED AS OF DATE:\s*(\d{8})",
        "fiscal_period_end":r"CONFORMED PERIOD OF REPORT:\s*(\d{8})",
        "cik":r"CENTRAL INDEX KEY:\s*(\d{1,10})",
    }.items():
        m=re.search(pattern,header,re.I)
        if m:
            raw=m.group(1)
            result[key]=f"{raw[:4]}-{raw[4:6]}-{raw[6:]}" if key!="cik" else raw.zfill(10)
    return result

def filing_metadata(path,soup):
    ticker=next((p.upper() for p in path.parts if p.upper() in TICKERS),"UNKNOWN")
    accession=next((p for p in path.parts if re.fullmatch(r"\d{10}-\d{2}-\d{6}",p)),"")
    # DEI facts are ix:nonNumeric elements; the concept is stored in the name attribute.
    cik=inline_fact_by_name(soup,"dei:EntityCentralIndexKey")
    period=iso_date(inline_fact_by_name(soup,"dei:DocumentPeriodEndDate"))
    header=sec_submission_header(path)
    cik=cik or header.get("cik")
    filed=header.get("filing_date")
    period=period or header.get("fiscal_period_end")
    return {"ticker":ticker,"issuer":{"AAPL":"Apple Inc.","MSFT":"Microsoft Corporation","NVDA":"NVIDIA Corporation"}.get(ticker,ticker),"cik":cik or None,"accession_number":accession or None,"filing_date":filed,"fiscal_period_end":period,"source_file":str(path)}

def primary_html_files():
    if not DATA_ROOT.exists(): raise FileNotFoundError(DATA_ROOT)
    return sorted(DATA_ROOT.glob("*/10-K/*/primary-document.html"))

def extract_item_sections(soup):
    """Extract target Items from body-level heading blocks, excluding the TOC.

    SEC filings commonly place the table of contents inside tables while actual
    Item headings are direct children of body. Collect sibling blocks until the
    next actual Item heading, retaining tables and Inline XBRL text in between.
    """
    body=soup.body
    if body is None:
        return {}
    item_heading=re.compile(r"^\s*item\s+(\d+[a-z]?)\s*[.\-–—:]?\s*.*$",re.I)
    blocks=[]
    for child in body.find_all(recursive=False):
        if not isinstance(child,Tag):
            continue
        heading_text=clean(child.get_text(" ",strip=True))
        match=item_heading.match(heading_text)
        if match and len(heading_text)<=300:
            blocks.append((child,match.group(1).upper(),heading_text))
    found={}
    for heading,item,heading_text in blocks:
        if item not in ITEMS or item in found:
            continue
        pieces=[heading_text]
        for sibling in heading.find_next_siblings():
            if not isinstance(sibling,Tag):
                continue
            sibling_text=clean(sibling.get_text(" ",strip=True))
            next_heading=item_heading.match(sibling_text)
            if next_heading and len(sibling_text)<=300:
                break
            if sibling_text:
                pieces.append(sibling_text)
        content=clean(" ".join(pieces))
        if len(content)>30:
            found[item]=content
    return found

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); files=primary_html_files()
    if not files: raise RuntimeError(f"No primary-document.html files under {DATA_ROOT}")
    manifest=[]; facts=[]; narratives=[]
    splitter=RecursiveCharacterTextSplitter(chunk_size=1800,chunk_overlap=250)
    for path in files:
        digest=sha256_file(path); raw=path.read_bytes(); soup=BeautifulSoup(raw,"lxml")
        meta=filing_metadata(path,soup); meta.update({"source_sha256":digest,"source_size_bytes":len(raw)})
        manifest.append(meta); filing_facts=extract_inline_facts(soup,meta,str(path)); facts.extend(filing_facts)
        for item,body in extract_item_sections(soup).items():
            doc=Document(page_content=body,metadata={"record_type":"narrative","section_item":item,"section_title":ITEMS[item],**{k:str(meta.get(k) or "") for k in ("ticker","issuer","cik","accession_number","filing_date","fiscal_period_end","source_file","source_sha256")}})
            narratives.extend(splitter.split_documents([doc]))
        if sha256_file(path)!=digest: raise RuntimeError(f"Source changed: {path}")
        print(f"{meta['ticker']} {meta['accession_number']}: {len(filing_facts)} inline facts")
    (OUTPUT/"filing_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    with (OUTPUT/"filing_manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
    with (OUTPUT/"xbrl_facts.jsonl").open("w",encoding="utf-8") as f:
        for fact in facts: f.write(json.dumps(fact,ensure_ascii=False)+"\n")
    fact_docs=[]
    for fact in facts:
        ctx=fact.get("context") or {}
        body=f"XBRL {fact['concept']} = {fact['value']} {fact.get('unit') or ''}; context={ctx}; issuer={fact['issuer']}; accession={fact['accession_number']}"
        md={"record_type":"xbrl_fact","fact_id":fact["fact_id"],"concept":fact["concept"],"value":str(fact["value"]),"unit":str(fact.get("unit") or ""),"scale":str(fact.get("scale") or ""),"decimals":str(fact.get("decimals") or ""),"context_ref":str(fact.get("context_ref") or ""),"context_period_start":str(ctx.get("period_start") or ""),"context_period_end":str(ctx.get("period_end") or ""),"context_instant":str(ctx.get("instant") or ""),"dimensions_json":json.dumps(ctx.get("dimensions",[])),"source_location":fact["source_location"]}
        md.update({k:str(fact.get(k) or "") for k in ("ticker","issuer","cik","accession_number","filing_date","fiscal_period_end","source_file")})
        fact_docs.append(Document(page_content=body,metadata=md))
    embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    if narratives: Chroma.from_documents(narratives,embeddings,collection_name=NARRATIVE_COLLECTION,persist_directory=str(VECTOR_DIR))
    if fact_docs: Chroma.from_documents(fact_docs,embeddings,collection_name=FACT_COLLECTION,persist_directory=str(VECTOR_DIR))
    summary={"filings":len(manifest),"narrative_chunks":len(narratives),"inline_xbrl_facts":len(facts),"fact_documents":len(fact_docs),"source_files_unchanged":True,"collections":[NARRATIVE_COLLECTION,FACT_COLLECTION],"warning":"Validate section extraction across all issuers and filing years; this lightweight parser is not a complete iXBRL conformance processor."}
    (OUTPUT/"ingestion_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
