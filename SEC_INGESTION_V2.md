# SEC 10-K / Inline XBRL ingestion v2

The legacy parser removed every HTML table before extracting text, which could discard financial statement content and Inline XBRL facts. It also mixed HTML and TXT files and selected the first nine entries. The v2 pipeline explicitly selects `primary-document.html` and reads source files without modifying them.

## Run in Python 3.11

```bash
python -m pip install -r requirements.txt
python -m pytest tests/test_sec_ingestion_v2.py -v
python 09_ingest_sec_v2.py
```

## Metadata extraction

- Issuer CIK is read from the Inline XBRL fact whose `name` attribute is `dei:EntityCentralIndexKey`.
- Fiscal period end is read from `dei:DocumentPeriodEndDate`, normalized to ISO format, with the SEC submission header's `CONFORMED PERIOD OF REPORT` as fallback.
- Filing date is read from the companion `full-submission.txt` SEC header's `FILED AS OF DATE`.
- The accession number is taken from the filing directory name.
- The companion submission file is read only; neither source file is modified.

## Narrative section extraction

The extractor uses direct children of the document `body` as candidate Item heading blocks. This avoids table-of-contents headings nested inside tables in the tested Workiva filings. It captures following sibling blocks until the next Item heading, retaining text and table content in the section. The target Items are 1, 1A, 7 and 8.

This is a structural heuristic, not a universal SEC filing parser. Validate heading structure across all issuers and filing years before relying on the extracted corpus.

## Outputs

- `sec_processed_v2/filing_manifest.json` and `filing_manifest.csv`: accession, issuer CIK, filing date, fiscal period, source path, SHA-256 and size.
- `sec_processed_v2/xbrl_facts.jsonl`: concept, value, raw value, unit, scale, decimals, context, dimensions and source location.
- `chroma_sec_v2`: separate `sec_narrative_v2` and `sec_xbrl_facts_v2` collections linked by accession/ticker/CIK/period metadata.

## Review gates

1. This is a lightweight Inline XBRL parser, not a complete iXBRL conformance processor. Validate continuations, transformations, nil facts, fraction facts, hidden facts and complex typed dimensions before claiming exhaustive coverage.
2. Confirm all nine filing metadata records are populated and match their SEC submission headers.
3. Inspect extracted Items 1, 1A, 7 and 8 for every filing. Check for table-of-contents contamination, missing tables and incorrect section boundaries.
4. Dimensions are retained as JSON strings in Chroma metadata; complete facts remain in JSONL.
5. The legacy Chroma baseline, existing benchmark, evaluation outputs and original filing files are not overwritten.
