# SEC 10-K / Inline XBRL ingestion v2

The legacy parser removed every HTML table before extracting text, which could discard financial statement content and Inline XBRL facts. It also mixed HTML and TXT files and selected the first nine entries. The v2 pipeline explicitly selects primary-document.html and reads each source in read-only mode.

## Run in Python 3.11
```bash
python -m pip install -r requirements.txt
python 09_ingest_sec_v2.py
python -m pytest tests/test_sec_ingestion_v2.py -q
```

## Outputs
- sec_processed_v2/filing_manifest.json and filing_manifest.csv: accession, CIK, issuer, filing date, fiscal period, source path, SHA-256 and size.
- sec_processed_v2/xbrl_facts.jsonl: concept, value, raw value, unit, scale, decimals, context, dimensions and source location.
- chroma_sec_v2: separate sec_narrative_v2 and sec_xbrl_facts_v2 collections linked by accession/ticker/CIK/period metadata.

## Review gates
1. This is a lightweight Inline XBRL parser, not a complete iXBRL conformance processor. Validate continuations, transformations, nil facts, fraction facts, hidden facts and complex typed dimensions before claiming exhaustive coverage.
2. Filing date and fiscal period are extracted from DEI/SEC fields when available. Missing values remain blank and must be checked against filing headers.
3. Item boundary extraction is heuristic; inspect extracted Items 1, 1A, 7 and 8 because table-of-contents entries and repeated headings can confuse boundaries.
4. Dimensions are retained as JSON strings in Chroma metadata; complete facts remain in JSONL.
5. The legacy Chroma baseline, existing benchmark, evaluation outputs and original filing files are not overwritten.
