# 03_baseline_rag.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load Chroma vector store
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_db = Chroma(persist_directory="./chroma_baseline_db", embedding_function=embeddings)

def vanilla_rag_retrieve(query: str, top_k: int = 5):
    """Simple linear semantic search without reranking or validation."""
    results = vector_db.similarity_search_with_score(query, k=top_k)
    return results

if __name__ == "__main__":
    # Test a tricky multi-hop / look-alike query
    test_query = "What specific risks did Apple report regarding foreign exchange rates and international supply chains?"
    
    print(f"\nQuery: {test_query}\n")
    matches = vanilla_rag_retrieve(test_query, top_k=3)
    
    for i, (doc, score) in enumerate(matches, 1):
        print(f"--- Chunk {i} (Distance Score: {score:.4f}) | Ticker: {doc.metadata.get('ticker')} ---")
        print(doc.page_content[:250] + "...\n")