import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# --- BEST-PRACTICE OFFLINE CONFIGURATION ---
# This path is confirmed to contain the 311-compatible models
MODEL_CACHE_DIR = r"C:\docling_dist-313\models_cache_311"

# Environment variables to force absolute offline behavior
os.environ["DOCLING_ARTIFACTS_PATH"] = MODEL_CACHE_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "*"

# Import Docling modules AFTER setting environment variables
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling_core.types.doc.base import ImageRefMode
from docling.chunking import HierarchicalChunker

# Professional Logging Setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("offline_parsing_311.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("OfflineParser311")

class IngestionEngine:
    """
    Best-Practice Offline Ingestion Engine for Python 3.11.
    Optimized for stability, speed, and structural chunking.
    """
    def __init__(self, use_ocr: bool = True, use_formula: bool = False):
        logger.info("Initializing IngestionEngine [OFFLINE MODE]")
        logger.info(f"Using 311 Model Cache: {MODEL_CACHE_DIR}")
        
        # 1. Setup Pipeline Options
        self.pipeline_options = PdfPipelineOptions()
        
        # WE DO NOT set artifacts_path manually here to avoid internal logic breakage.
        # Docling will find them via DOCLING_ARTIFACTS_PATH env var automatically.
        
        # 2. Configure Features
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.do_code_enrichment = True
        
        # 3. Handle Formula Enrichment (Disabled by default for stability)
        self.pipeline_options.do_formula_enrichment = use_formula
        
        # 4. Configure RapidOCR (Standard for 311 offline)
        if use_ocr:
            self.pipeline_options.do_ocr = True
            # We use default RapidOcrOptions to let Docling find files via the Artifacts Path
            self.pipeline_options.ocr_options = RapidOcrOptions()
            logger.info("OCR Enabled: RapidOCR")
        else:
            self.pipeline_options.do_ocr = False
            logger.info("OCR Disabled")

        # 5. Initialize Converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        
        # 6. Setup Hierarchical Chunker (Best for RAG/Offline context)
        self.chunker = HierarchicalChunker()

    def process(self, pdf_path: str, output_root: str, skip_start: int = 0, skip_end: int = 0):
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

        # Create Output Structure
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        job_dir = Path(output_root) / f"{pdf_path.stem}_parsed_{timestamp}"
        img_dir = job_dir / "images"
        table_dir = job_dir / "tables"
        
        for d in [job_dir, img_dir, table_dir]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing File: {pdf_path.name}")
        logger.info(f"Target Output: {job_dir}")
        
        start_time = time.time()

        # Step 1: Initial Parsing (Get Metadata)
        logger.info("Parsing document structure...")
        conv_res = self.converter.convert(pdf_path)
        total_pages = len(conv_res.pages)
        
        # Calculate Range
        start_p = skip_start + 1
        end_p = total_pages - skip_end
        if start_p > end_p or end_p < 1:
            raise ValueError(f"Invalid range: Page {start_p} to {end_p}. Doc has {total_pages} pages.")

        logger.info(f"Page Range Selected: {start_p} to {end_p}")

        # Step 2: Refined Parsing for Page Range
        # Note: Re-parsing ensures the Markdown and Chunks are perfectly synced
        logger.info(f"Extracting content from pages {start_p}-{end_p}...")
        range_res = self.converter.convert(pdf_path, page_range=(start_p, end_p))
        doc = range_res.document

        # Step 3: Save Images
        img_count = 0
        for i, element in enumerate(doc.pictures):
            if element.image:
                img_name = f"image_{img_count+1:03d}.png"
                element.image.pil_image.save(img_dir / img_name)
                element.image.uri = Path("images") / img_name
                img_count += 1
        logger.info(f"Extracted {img_count} images.")

        # Step 4: Save Tables
        table_count = 0
        for i, table in enumerate(doc.tables):
            csv_path = table_dir / f"table_{table_count+1:03d}.csv"
            table.export_to_dataframe().to_csv(csv_path, index=False)
            table_count += 1
        logger.info(f"Extracted {table_count} tables.")

        # Step 5: Export Markdown
        md_path = job_dir / f"{pdf_path.stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(doc.export_to_markdown(image_mode=ImageRefMode.REFERENCED))
        logger.info("Markdown saved.")

        # Step 6: Structural Chunking
        logger.info("Generating structural chunks...")
        chunks_data = []
        for i, chunk in enumerate(self.chunker.chunk(dl_doc=doc)):
            page_numbers = set()
            if hasattr(chunk.meta, 'doc_items'):
                for item in chunk.meta.doc_items:
                    if hasattr(item, 'prov') and item.prov:
                        for p in item.prov:
                            page_numbers.add(p.page_no)
            
            chunks_data.append({
                "chunk_id": i + 1,
                "text": chunk.text,
                "metadata": {
                    "pages": sorted(list(page_numbers)),
                    "headings": getattr(chunk.meta, 'headings', []),
                    "source": pdf_path.name
                }
            })

        with open(job_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        duration = time.time() - start_time
        logger.info(f"COMPLETED successfully in {duration:.2f}s")
        logger.info(f"Result Directory: {job_dir}")
        return str(job_dir)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DOCLING OFFLINE BEST-PRACTICE PARSER (311 Cache)")
    print("="*60)
    
    engine = None
    try:
        f_path = input("1. Enter PDF file path: ").strip().strip('"')
        o_root = input("2. Enter Output storage location: ").strip().strip('"')
        s_start = int(input("3. Pages to skip from START (default 0): ") or 0)
        s_end = int(input("4. Pages to skip from END (default 0): ") or 0)
        
        use_f = input("5. Enable Formula Enrichment? (High Accuracy, but slower) [y/N]: ").lower().strip() == 'y'
        
        # Initialize Engine
        engine = IngestionEngine(use_ocr=True, use_formula=use_f)
        engine.process(f_path, o_root, s_start, s_end)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        print(f"\n[ERROR]: {e}")
    finally:
        # Explicitly delete the engine to trigger destructors 
        # while the logging module is still active.
        if engine:
            logger.info("Cleaning up resources...")
            del engine
            import gc
            gc.collect()
        print("\nExiting script.")
