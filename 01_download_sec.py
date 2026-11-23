# 01_download_sec.py
from sec_edgar_downloader import Downloader
import os

# SEC EDGAR requires: "Sample Company Name AdminContact@<sample company domain>.com"
USER_AGENT = "MScDataScienceResearch student_research@university.edu"

dl = Downloader("MyResearchOrg", "student_research@university.edu", "./sec_data")

tickers = ["AAPL", "MSFT", "NVDA"]

print("Starting SEC 10-K downloads...")
for ticker in tickers:
    print(f"Downloading 10-Ks for {ticker}...")
    # Downloads the 3 most recent 10-K filings for each company
    dl.get("10-K", ticker, limit=3, download_details=True)

print("Download complete! Files saved in ./sec_data")