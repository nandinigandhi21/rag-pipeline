import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# --- BEST-PRACTICE OFFLINE CONFIGURATION ---
# Points to the model distribution folder
MODEL_CACHE_DIR = r"C:\docling_dist-313\models_cache_311"

os.environ["DOCLING_ARTIFACTS_PATH"] = MODEL_CACHE_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "*"

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling_core.types.doc.base import ImageRefMode
from docling.chunking import HybridChunker

# Professional Logging Setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hybrid_parsing.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("HybridParser")

class IngestionEngine:
    """
    Offline Ingestion Engine using Hybrid Chunking.
    Best for certificates, forms, and technical sheets where labels/values must stay together.
    """
    def __init__(self, use_ocr: bool = True, use_formula: bool = False):
        logger.info("Initializing Hybrid IngestionEngine [OFFLINE MODE]")
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.do_code_enrichment = True
        self.pipeline_options.do_formula_enrichment = use_formula
        
        if use_ocr:
            self.pipeline_options.do_ocr = True
            self.pipeline_options.ocr_options = RapidOcrOptions()
        else:
            self.pipeline_options.do_ocr = False

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        
        # Setup Hybrid Chunker with Local Tokenizer
        tokenizer_path = str(Path(MODEL_CACHE_DIR) / "docling-project--CodeFormulaV2")
        logger.info(f"Using Tokenizer: {tokenizer_path}")
        self.chunker = HybridChunker(tokenizer=tokenizer_path, max_tokens=512, merge_peers=True)

    def process(self, pdf_path: str, output_root: str, skip_start: int = 0, skip_end: int = 0):
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        job_dir = Path(output_root) / f"{pdf_path.stem}_hybrid_{timestamp}"
        img_dir = job_dir / "images"
        table_dir = job_dir / "tables"
        
        for d in [job_dir, img_dir, table_dir]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing: {pdf_path.name}")
        start_time = time.time()

        # Step 1: Parsing
        conv_res = self.converter.convert(pdf_path)
        total_pages = len(conv_res.pages)
        start_p = skip_start + 1
        end_p = total_pages - skip_end
        
        logger.info(f"Page Range: {start_p} to {end_p}")
        range_res = self.converter.convert(pdf_path, page_range=(start_p, end_p))
        doc = range_res.document

        # Step 2: Save Assets
        img_count = 0
        for i, element in enumerate(doc.pictures):
            if element.image:
                img_name = f"image_{img_count+1:03d}.png"
                element.image.pil_image.save(img_dir / img_name)
                element.image.uri = Path("images") / img_name
                img_count += 1

        for i, table in enumerate(doc.tables):
            csv_path = table_dir / f"table_{i+1:03d}.csv"
            table.export_to_dataframe().to_csv(csv_path, index=False)

        # Step 3: Markdown & Chunks
        md_path = job_dir / f"{pdf_path.stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(doc.export_to_markdown(image_mode=ImageRefMode.REFERENCED))

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
                    "strategy": "hybrid"
                }
            })

        with open(job_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        duration = time.time() - start_time
        logger.info(f"COMPLETED in {duration:.2f}s. Saved to {job_dir}")

if __name__ == "__main__":
    print("\nDOCLING OFFLINE: HYBRID PARSER")
    engine = None
    try:
        f_path = input("PDF Path: ").strip().strip('"')
        o_root = input("Output Location: ").strip().strip('"')
        s_start = int(input("Skip Start: ") or 0)
        s_end = int(input("Skip End: ") or 0)
        use_f = input("Enable Formulas? [y/N]: ").lower().strip() == 'y'
        
        engine = IngestionEngine(use_ocr=True, use_formula=use_f)
        engine.process(f_path, o_root, s_start, s_end)
    except Exception as e:
        logger.error(f"FATAL: {e}")
    finally:
        if engine:
            del engine
            import gc
            gc.collect()
        print("\nExiting script.")
