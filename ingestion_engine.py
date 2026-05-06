import os
import time
import json
import logging
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pypdfium2 as pdfium 

# Docling Core Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, TableFormerMode
from docling_core.types.doc.labels import DocItemLabel
from docling.chunking import HierarchicalChunker

# 1. OFFLINE CONFIGURATION (Updated for your 3.11/313 setup)
BASE_PATH = r"C:\docling_dist-313\models_cache"
os.environ["DOCLING_ARTIFACTS_PATH"] = BASE_PATH
os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngestionEngine:
    def __init__(self, output_root: str = "rag_storage"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 2. STABLE CONFIGURATION FOR 3.11
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.artifacts_path = BASE_PATH
        
        # Use Accelerator Options for threading
        self.pipeline_options.accelerator_options.num_threads = 2
        
        # High Accuracy Features
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        self.pipeline_options.do_formula_enrichment = True 
        self.pipeline_options.do_code_enrichment = True
        
        # Stable OCR (RapidOCR instead of EasyOCR)
        self.pipeline_options.do_ocr = True
        self.pipeline_options.ocr_options = RapidOcrOptions()
        self.pipeline_options.images_scale = 2.0 # Memory safe scale

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        self.chunker = HierarchicalChunker()

    def process_pdf(self, pdf_path: str, page_range: Optional[Tuple[int, int]] = None, block_size: int = 15) -> str:
        """
        Processes PDF sequentially to prevent RAM crashes (bad_alloc).
        Returns the path to the resulting JSON chunks.
        """
        pdf_path = Path(pdf_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        job_dir = self.output_root / f"{pdf_path.stem}_{timestamp}"
        job_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"--- STARTING SEQUENTIAL INGESTION: {pdf_path.name} ---")
        start_time = time.time()
        
        # Resolve total pages
        pdf_doc = pdfium.PdfDocument(str(pdf_path))
        total_pdf_pages = len(pdf_doc)
        
        start_page = page_range[0] if page_range else 1
        end_page = min(page_range[1], total_pdf_pages) if page_range else total_pdf_pages
        
        all_chunks = []
        current_p = start_page
        
        while current_p <= end_page:
            block_end = min(current_p + block_size - 1, end_page)
            target = (current_p, block_end)
            logger.info(f"Processing pages {target}...")
            
            try:
                # 3. Block Conversion
                result = self.converter.convert(pdf_path, page_range=target)
                
                # 4. Enriched Metadata Chunking
                for i, chunk in enumerate(self.chunker.chunk(dl_doc=result.document)):
                    labels = {str(item.label) for item in getattr(chunk.meta, 'doc_items', [])}
                    
                    # Filter Noise (Headers/Footers)
                    if any(l in [str(DocItemLabel.PAGE_HEADER), str(DocItemLabel.PAGE_FOOTER)] for l in labels):
                        continue

                    all_chunks.append({
                        "text": chunk.text,
                        "metadata": {
                            "headings": chunk.meta.headings if hasattr(chunk.meta, 'headings') else [],
                            "labels": list(labels),
                            "page_range": list(target),
                            "char_count": len(chunk.text)
                        }
                    })

                # Manual cleanup to free RAM for next block
                del result
                gc.collect()
                current_p += block_size

            except Exception as e:
                logger.error(f"Error at page {current_p}: {e}")
                break

        # 5. Export results
        json_path = job_dir / "chunks.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        # Save MD for reference
        # (Note: Sequential MD export is limited to the last chunk, 
        # normally you'd re-parse the full doc for a single MD, 
        # but here we prioritize chunking speed/safety).
        
        duration = time.time() - start_time
        logger.info(f"COMPLETED: {len(all_chunks)} chunks saved in {duration:.2f}s")
        return str(json_path)

def resolve_page_range(pdf_path: str, skip_start: int, skip_end: int) -> Optional[Tuple[int, int]]:
    """Helper for GUI to calculate ranges."""
    pdf = pdfium.PdfDocument(pdf_path)
    total = len(pdf)
    start = 1 + skip_start
    end = total - skip_end
    if start > end: return None
    return (start, end)
