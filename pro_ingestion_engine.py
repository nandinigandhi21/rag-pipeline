import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pypdfium2 as pdfium 

# Docling Core Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, TableFormerMode
from docling_core.types.doc.labels import DocItemLabel
from docling.chunking import HierarchicalChunker

# 1. OFFLINE CONFIGURATION
BASE_PATH = r"C:\docling_dist-313\models_cache"
os.environ["DOCLING_ARTIFACTS_PATH"] = BASE_PATH
os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SinglePassEngine:
    """
    High-Performance Engine that processes the entire PDF in a single call.
    Requires significant RAM for large documents.
    """
    def __init__(self, output_root: str = "rag_storage"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 2. AGGRESSIVE PERFORMANCE CONFIGURATION
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.artifacts_path = BASE_PATH
        
        # Threads for multi-core CPUs
        self.pipeline_options.accelerator_options.num_threads = 4
        
        # Internal C++ Batching (Optimizes memory throughput)
        self.pipeline_options.ocr_batch_size = 32
        self.pipeline_options.layout_batch_size = 16
        
        # High Accuracy Features
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        self.pipeline_options.do_formula_enrichment = True 
        self.pipeline_options.do_code_enrichment = True
        
        # Stable OCR
        self.pipeline_options.do_ocr = True
        self.pipeline_options.ocr_options = RapidOcrOptions()
        self.pipeline_options.images_scale = 2.0 

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        self.chunker = HierarchicalChunker()

    def process_pdf(self, pdf_path: str, page_range: Optional[Tuple[int, int]] = None) -> str:
        pdf_path = Path(pdf_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        job_dir = self.output_root / f"{pdf_path.stem}_{timestamp}"
        job_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"--- STARTING SINGLE-PASS INGESTION: {pdf_path.name} ---")
        start_time = time.time()

        try:
            # 3. DIRECT CONVERSION (Single Call)
            result = self.converter.convert(pdf_path, page_range=page_range)
            
            # 4. ENRICHED CHUNKING
            chunks_data = []
            for i, chunk in enumerate(self.chunker.chunk(dl_doc=result.document)):
                labels = {str(item.label) for item in getattr(chunk.meta, 'doc_items', [])}
                
                # Filter Headers/Footers
                if any(l in [str(DocItemLabel.PAGE_HEADER), str(DocItemLabel.PAGE_FOOTER)] for l in labels):
                    continue

                chunks_data.append({
                    "text": chunk.text,
                    "metadata": {
                        "headings": chunk.meta.headings if hasattr(chunk.meta, 'headings') else [],
                        "depth": len(chunk.meta.headings) if hasattr(chunk.meta, 'headings') else 0,
                        "labels": list(labels),
                        "char_count": len(chunk.text)
                    }
                })

            # 5. EXPORT
            json_path = job_dir / "full_chunks.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)

            # Export full Markdown
            with open(job_dir / f"{pdf_path.stem}.md", "w", encoding="utf-8") as f:
                f.write(result.document.export_to_markdown())

            duration = time.time() - start_time
            logger.info(f"COMPLETED: {len(chunks_data)} chunks saved in {duration:.2f}s")
            return str(json_path)

        except Exception as e:
            logger.error(f"CRITICAL FAILURE: {e}")
            raise e

def resolve_page_range(pdf_path: str, skip_start: int, skip_end: int) -> Optional[Tuple[int, int]]:
    pdf = pdfium.PdfDocument(pdf_path)
    total = len(pdf)
    start = 1 + skip_start
    end = total - skip_end
    if start > end: return None
    return (start, end)
