# 05_hybrid_reranker.py
import re
from typing import List, Optional, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class HybridDynamicReranker:
    def __init__(
        self,
        persist_directory: str = "./chroma_baseline_db",
        embed_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        # 1. Load existing ChromaDB dense index
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        self.vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )

        # 2. Extract stored documents from ChromaDB to build the in-memory BM25 lexical index
        raw_data = self.vector_db.get(include=["documents", "metadatas"])
        self.all_docs: List[Document] = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw_data["documents"], raw_data["metadatas"])
        ]

        tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. Initialize CPU-friendly Stage-2 Cross-Encoder
        self.cross_encoder = CrossEncoder(reranker_model, device="cpu")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def compute_dynamic_k(self, query: str) -> float:
        """
        Novel Heuristic: Adjusts RRF penalty parameter k dynamically based on
        uppercase financial tickers/acronyms and numeric tokens in the query.
        """
        acronyms_and_numbers = re.findall(r"\b(?:[A-Z]{2,5}|\d+(?:\.\d+)?)\b", query)
        k_dyn = 60.0 - (3.0 * len(acronyms_and_numbers))
        return max(10.0, min(60.0, k_dyn))

    def retrieve_and_rerank(
        self,
        query: str,
        target_ticker: Optional[str] = None,
        stage1_k: int = 25,
        final_top_k: int = 5,
    ) -> List[Document]:
        # --- STAGE 1A: Dense Vector Retrieval ---
        search_kwargs = {"k": stage1_k}
        if target_ticker in ["AAPL", "MSFT", "NVDA"]:
            search_kwargs["filter"] = {"ticker": target_ticker}

        dense_docs = self.vector_db.similarity_search(query, **search_kwargs)

        # --- STAGE 1B: Sparse BM25 Lexical Retrieval ---
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Apply ticker filter to BM25 candidates if specified
        scored_indices = []
        for idx, score in enumerate(bm25_scores):
            if target_ticker in ["AAPL", "MSFT", "NVDA"]:
                if self.all_docs[idx].metadata.get("ticker") != target_ticker:
                    continue
            scored_indices.append((idx, score))

        scored_indices.sort(key=lambda x: x[1], reverse=True)
        sparse_docs = [self.all_docs[idx] for idx, _ in scored_indices[:stage1_k]]

        # --- DYNAMIC RECIPROCAL RANK FUSION (RRF) ---
        k_dyn = self.compute_dynamic_k(query)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs, start=1):
            doc_key = doc.page_content[:200]
            doc_map[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (1.0 / (k_dyn + rank))

        for rank, doc in enumerate(sparse_docs, start=1):
            doc_key = doc.page_content[:200]
            doc_map[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (1.0 / (k_dyn + rank))

        fused_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:stage1_k]
        candidate_docs = [doc_map[key] for key in fused_keys]

        if not candidate_docs:
            return []

        # --- STAGE 2: Cross-Encoder Reranking ---
        pairs = [[query, doc.page_content] for doc in candidate_docs]
        ce_scores = self.cross_encoder.predict(pairs)

        reranked = sorted(
            zip(candidate_docs, ce_scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        top_docs = []
        for doc, ce_score in reranked[:final_top_k]:
            # Attach dynamic RRF and Cross-Encoder scores to metadata for thesis logging
            updated_meta = dict(doc.metadata)
            updated_meta["dynamic_rrf_k"] = k_dyn
            updated_meta["reranker_score"] = round(float(ce_score), 4)
            top_docs.append(Document(page_content=doc.page_content, metadata=updated_meta))

        return top_docs