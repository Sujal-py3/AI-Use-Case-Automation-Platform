import time
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# Import helper for logging
from src.utils import setup_logger

logger = setup_logger("retrieval")

class Retriever:
    def __init__(self, vector_db: Chroma, all_documents: List[Document]):
        """
        Sets up the Hybrid Retriever.
        We need 'all_documents' to build the BM25 index on the fly.
        """
        self.vector_db = vector_db
        self.all_documents = all_documents
        self.ensemble_retriever = None
        
        # Initialize the retrievers
        self._setup_retrievers()

    def _setup_retrievers(self):
        try:
            # 1. Vector Retriever (Semantic Search)
            # search_kwargs={"k": 5} means get top 5 results
            vector_retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
            
            # 2. Keyword Retriever (BM25) - Good for exact matches
            bm25_retriever = BM25Retriever.from_documents(self.all_documents)
            bm25_retriever.k = 5
            
            # 3. Hybrid (Ensemble)
            # We weight them 50/50. 
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
            logger.info("Hybrid Retriever (Vector + BM25) initialized.")
            
        except Exception as e:
            logger.error(f"Failed to setup retrievers: {e}")
            self.ensemble_retriever = None

    def query(self, user_query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieves relevant documents for the query.
        Includes GUARDRAILS:
        - Latency logging
        - (Optional) Score thresholding could be done if we used raw search, 
          but Ensemble doesn't return scores easily. 
          So we'll rely on the top_k ranking from the ensemble.
        """
        if not self.ensemble_retriever:
            logger.error("Retriever not initialized!")
            return []

        start_time = time.time()
        logger.info(f"Retrieving context for query: '{user_query}'")

        try:
            # Get documents
            # Note: EnsembleRetriever.invoke(query) returns a list of Documents
            docs = self.ensemble_retriever.invoke(user_query)
            
            # Since Ensemble doesn't return scores directly in the standard invoke,
            # we just slice the list to the requested top_k.
            # (The ensemble already sorts them by weighted rank).
            relevant_docs = docs[:top_k]
            
            end_time = time.time()
            logger.info(f"Retrieval finished in {end_time - start_time:.4f}s. Found {len(relevant_docs)} docs.")
            
            # Log what we found for observability (just the sources)
            sources = [d.metadata.get('source', 'unknown') for d in relevant_docs]
            logger.info(f"Retrieved sources: {sources}")
            
            return relevant_docs

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
