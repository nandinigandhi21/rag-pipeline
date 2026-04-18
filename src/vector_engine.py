import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configuration
os.environ["HF_HUB_OFFLINE"] = "1"
# Ensure this folder exists or change to where you saved BGE-v1.5
MODEL_PATH = r"C:/docling_dist/models_cache/bge-small-en-v1.5"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorEngine:
    """
    Professional Vector Engine using ChromaDB and BGE-v1.5 for Retrieval.
    """
    def __init__(self, db_path: str = "rag_storage/chroma_db"):
        self.db_path = db_path
        
        # Initialize Embedding Model (Offline)
        logger.info(f"Loading embedding model from: {MODEL_PATH}")
        # Note: If path doesn't exist, this will fail in offline mode.
        self.model = SentenceTransformer(MODEL_PATH)
        
        # Initialize ChromaDB (Persistent)
        self.client = chromadb.PersistentClient(path=self.db_path)
        
    def populate_from_json(self, chunks_json_path: str, collection_name: str):
        with open(chunks_json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # L2 Normalization handled by cosine space
        )

        documents = []
        metadatas = []
        ids = []
        
        logger.info(f"Preparing {len(chunks)} chunks for vectorization...")
        
        for chunk in chunks:
            # PROFESSIONAL STRATEGY: Augmented Context
            # We "bake" the headers into the text so the vector understands the context.
            header_context = " > ".join(chunk["metadata"]["headings"])
            labels_context = ", ".join(chunk["metadata"]["labels"])
            augmented_text = f"CONTEXT: [{header_context}] TYPE: [{labels_context}] CONTENT: {chunk['text']}"
            
            documents.append(augmented_text)
            
            # Prepare metadata (ChromaDB requires flat dicts)
            flat_meta = {
                "source": chunk["metadata"]["source"],
                "doc_title": chunk["metadata"]["doc_title"],
                "headings": json.dumps(chunk["metadata"]["headings"]),
                "page_numbers": json.dumps(chunk["metadata"]["page_numbers"]),
                "labels": json.dumps(chunk["metadata"]["labels"]),
                "chunk_id": chunk["id"]
            }
            metadatas.append(flat_meta)
            ids.append(chunk["id"])

        # Batch add to Chroma (Chroma handles the model call if we passed an embedding function,
        # but here we generate them manually for maximum professional control).
        logger.info("Generating embeddings (Batching enabled)...")
        embeddings = self.model.encode(documents, show_progress_bar=True, batch_size=32).tolist()
        
        logger.info("Updating Vector Database...")
        collection.add(
            embeddings=embeddings,
            documents=documents, # We store the augmented text
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Vector Store updated successfully with collection: {collection_name}")

    def query(self, collection_name: str, query_text: str, n_results: int = 5):
        collection = self.client.get_collection(name=collection_name)
        
        # Encode the query
        query_embedding = self.model.encode([query_text]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results

if __name__ == "__main__":
    # Example usage (assuming ingestion was run and model files are present)
    # Change collection name based on your PDF
    v_engine = VectorEngine()
    
    # Example Query
    # results = v_engine.query("ResNet_Paper", "What is identity mapping?")
    # print(results)
