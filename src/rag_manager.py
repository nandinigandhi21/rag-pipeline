import argparse
import sys
import logging
from pathlib import Path
from ingestion_engine import IngestionEngine
from vector_engine import VectorEngine
from generation_engine import GenerationEngine

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Professional Offline RAG Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Process a PDF into layout-aware chunks and index them")
    ingest_parser.add_argument("pdf", help="Path to the PDF file")
    ingest_parser.add_argument("--collection", required=True, help="Collection name for Vector DB")

    # Command: query
    query_parser = subparsers.add_parser("query", help="Ask a question about your indexed documents")
    query_parser.add_argument("--collection", required=True, help="Collection name to search in")
    query_parser.add_argument("--text", required=True, help="The question/query text")
    query_parser.add_argument("--top", type=int, default=4, help="Number of chunks to use as context")

    args = parser.parse_args()

    if args.command == "ingest":
        try:
            # 1. Ingestion Phase
            ingestor = IngestionEngine()
            chunks_path = ingestor.process_pdf(args.pdf)
            
            # 2. Vectorization Phase
            logger.info("Initializing Vector Engine...")
            vector_engine = VectorEngine()
            vector_engine.populate_from_json(chunks_path, args.collection)
            
            print(f"\nSUCCESS: Document indexed in collection '{args.collection}'")
            print(f"You can now run: python rag_manager.py query --collection {args.collection} --text \"{args.text if hasattr(args, 'text') else 'Your question'}\"")
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            sys.exit(1)

    elif args.command == "query":
        try:
            # 1. Retrieval Phase
            logger.info("Retrieving relevant context from Vector DB...")
            vector_engine = VectorEngine()
            results = vector_engine.query(args.collection, args.text, n_results=args.top)
            
            # Extract just the text from the search results
            context_chunks = results['documents'][0]
            
            # 2. Generation Phase
            logger.info("Initializing Generation Engine (LLM)...")
            gen_engine = GenerationEngine()
            answer = gen_engine.generate_answer(args.text, context_chunks)
            
            # 3. Professional Display
            print("\n" + "="*50)
            print(f"QUESTION: {args.text}")
            print("="*50)
            print(f"\nANSWER:\n{answer}")
            print("\n" + "="*50)
            print("SOURCES USED:")
            for i, meta in enumerate(results['metadatas'][0]):
                print(f"- {meta['source']} (Pages: {meta['page_numbers']})")
            print("="*50)
                
        except Exception as e:
            logger.error(f"Query failed: {e}")
            sys.exit(1)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
