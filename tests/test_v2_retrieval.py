from sec_v2_retriever import SECv2Retriever

retriever = SECv2Retriever()

tests = [
    {
        "name": "AAPL 2024",
        "ticker": "AAPL",
        "accession": "0000320193-24-000123",
        "period": "2024-09-28",
        "query": "Apple revenue net sales and financial performance",
        "section": "7"
    },
    {
        "name": "AAPL 2025",
        "ticker": "AAPL",
        "accession": "0000320193-25-000079",
        "period": "2025-09-27",
        "query": "Apple revenue net sales and financial performance",
        "section": "7"
    },
    {
        "name": "MSFT 2025",
        "ticker": "MSFT",
        "accession": "0000950170-25-100235",
        "period": "2025-06-30",
        "query": "Microsoft cloud revenue and operating performance",
        "section": "7"
    },
    {
        "name": "MSFT 2026",
        "ticker": "MSFT",
        "accession": "0001193125-26-323660",
        "period": "2026-06-30",
        "query": "Microsoft cloud revenue and operating performance",
        "section": "7"
    },
    {
        "name": "NVDA 2024",
        "ticker": "NVDA",
        "accession": "0001045810-24-000029",
        "period": "2024-01-28",
        "query": "NVIDIA revenue data center and financial performance",
        "section": "7"
    },
    {
        "name": "NVDA 2026",
        "ticker": "NVDA",
        "accession": "0001045810-26-000021",
        "period": "2026-01-25",
        "query": "NVIDIA revenue data center and financial performance",
        "section": "7"
    }
]

all_passed = True

for test in tests:
    print("\n" + "=" * 65)
    print("TEST:", test["name"])

    result = retriever.retrieve(
        query=test["query"],
        ticker=test["ticker"],
        accession_number=test["accession"],
        fiscal_period_end=test["period"],
        section_item=test["section"],
        k=5
    )

    for collection_name in ["narrative", "facts"]:
        docs = result[collection_name]

        accessions = {
            doc.metadata.get("accession_number")
            for doc in docs
        }

        periods = {
            doc.metadata.get("fiscal_period_end")
            for doc in docs
        }

        print(f"\n{collection_name.upper()}")
        print("Documents returned:", len(docs))
        print("Accessions:", accessions)
        print("Fiscal periods:", periods)

        valid = (
            len(docs) > 0
            and accessions == {test["accession"]}
            and periods == {test["period"]}
        )

        print("Result:", "PASS" if valid else "FAIL")

        if not valid:
            all_passed = False

print("\n" + "=" * 65)
print("OVERALL:", "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")