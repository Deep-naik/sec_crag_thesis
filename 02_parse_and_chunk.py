# 02_parse_and_chunk.py
import os
import glob
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def parse_html_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")
        
    # Remove script and style elements
    for element in soup(["script", "style", "table"]):
        element.decompose()
        
    text = soup.get_text(separator=" ")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)

def load_and_chunk_filings(base_dir="./sec_data"):
    # Target standard filing document paths produced by sec-edgar-downloader
    html_files = glob.glob(f"{base_dir}/sec-edgar-filings/**/*.html", recursive=True) + \
                 glob.glob(f"{base_dir}/sec-edgar-filings/**/*.txt", recursive=True)
    
    docs = []
    print(f"Found {len(html_files)} files. Parsing text...")

    for file_path in html_files[:9]: # 3 companies x 3 years
        # Extract basic metadata from folder structure
        parts = file_path.split(os.sep)
        ticker = "UNKNOWN"
        for idx, part in enumerate(parts):
            if part == "sec-edgar-filings" and idx + 1 < len(parts):
                ticker = parts[idx + 1]
                break
                
        cleaned_text = parse_html_file(file_path)
        
        # Keep manageable length for initial testing
        if len(cleaned_text) > 1000:
            docs.append(Document(
                page_content=cleaned_text[:150000],  # Take first ~30k tokens for testing
                metadata={"ticker": ticker, "source": file_path}
            ))
            print(f"Processed filing for {ticker}")

    # Chunking: 512 tokens (~2000 chars), overlap 64 tokens (~250 chars)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunked_docs = text_splitter.split_documents(docs)
    print(f"Total chunks generated: {len(chunked_docs)}")
    return chunked_docs

def build_vector_database(chunks):
    print("Initializing embedding model (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    print("Building ChromaDB baseline vector index...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_baseline_db"
    )
    print("ChromaDB vector store successfully created at ./chroma_baseline_db")
    return vector_db

if __name__ == "__main__":
    chunks = load_and_chunk_filings()
    build_vector_database(chunks)