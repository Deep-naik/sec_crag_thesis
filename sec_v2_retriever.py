"""Metadata-constrained retrieval over the v2 narrative and XBRL collections."""
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class SECv2Retriever:
    def __init__(self, persist_directory="chroma_sec_v2", embed_model="BAAI/bge-small-en-v1.5"):
        emb=HuggingFaceEmbeddings(model_name=embed_model)
        self.narrative=Chroma(persist_directory=persist_directory,collection_name="sec_narrative_v2",embedding_function=emb)
        self.facts=Chroma(persist_directory=persist_directory,collection_name="sec_xbrl_facts_v2",embedding_function=emb)

    @staticmethod
    def _where(ticker=None, accession_number=None, cik=None, fiscal_period_end=None, filing_date=None, section_item=None):
        clauses=[]
        for key,value in (("ticker",ticker),("accession_number",accession_number),("cik",cik),("fiscal_period_end",fiscal_period_end),("filing_date",filing_date),("section_item",section_item)):
            if value is not None: clauses.append({key:str(value)})
        if not clauses: return None
        return clauses[0] if len(clauses)==1 else {"$and":clauses}

    def retrieve(self, query, ticker=None, accession_number=None, cik=None, fiscal_period_end=None, filing_date=None, section_item=None, k=5):
        where=self._where(ticker,accession_number,cik,fiscal_period_end,filing_date,section_item)
        kwargs={"k":k}
        if where: kwargs["filter"]=where
        narrative=self.narrative.similarity_search(query,**kwargs)
        # Facts intentionally share issuer/accession/date filters; section_item is narrative-only.
        fact_where=self._where(ticker,accession_number,cik,fiscal_period_end,filing_date)
        fact_kwargs={"k":k}
        if fact_where: fact_kwargs["filter"]=fact_where
        facts=self.facts.similarity_search(query,**fact_kwargs)
        return {"narrative":narrative,"facts":facts}
