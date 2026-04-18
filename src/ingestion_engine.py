import os
import time
import json
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.labels import DocItemLabel
from docling.chunking import HierarchicalChunker

# Configuration
BASE_PATH = r"C:/docling_dist/models_cache"
os.environ["DOCLING_ARTIFACTS_PATH"] = BASE_PATH
os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngestionEngine:
    """
    Professional Ingestion Engine with Page Filtering and Header/Footer removal.
    """
    def __init__(self, output_root: str = "rag_storage"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize Pipeline Options
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.artifacts_path = BASE_PATH
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.do_formula_enrichment = True 
        self.pipeline_options.do_code_enrichment = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        self.pipeline_options.do_ocr = True
        self.pipeline_options.ocr_options = EasyOcrOptions() 
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)
            }
        )
        self.chunker = HierarchicalChunker()

    def process_pdf(self, pdf_path: str, page_range: Optional[Tuple[int, int]] = None) -> str:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Create structured output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        job_dir = self.output_root / f"{pdf_path.stem}_{timestamp}"
        img_dir = job_dir / "images"
        table_dir = job_dir / "tables"
        
        job_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(exist_ok=True)
        table_dir.mkdir(exist_ok=True)

        logger.info(f"Starting ingestion for: {pdf_path.name} (Range: {page_range})")
        start_time = time.time()

        # 1. Convert PDF with Page Range
        # Note: convert_all is used to support page_range directly
        conv_res_iter = self.converter.convert_all(
            [pdf_path], 
            page_range=page_range if page_range else (1, os.sys.maxsize)
        )
        result = next(conv_res_iter)
        
        # 2. Extract and Save Pictures
        for i, element in enumerate(result.document.pictures):
            if element.image:
                img_name = f"fig_{i+1:03d}.png"
                element.image.pil_image.save(img_dir / img_name)
                element.image.uri = Path("images") / img_name
        
        # 3. Extract and Save Tables
        for i, table in enumerate(result.document.tables):
            csv_path = table_dir / f"table_{i+1:03d}.csv"
            table.export_to_dataframe().to_csv(csv_path, index=False)

        # 4. Save High-Quality Markdown
        md_path = job_dir / f"{pdf_path.stem}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.document.export_to_markdown(image_mode=ImageRefMode.REFERENCED))

        # 5. Perform Enriched Hierarchical Chunking (Filtering Headers/Footers)
        logger.info("Performing hierarchical chunking with Header/Footer filtering...")
        chunks_data = []
        for i, chunk in enumerate(self.chunker.chunk(dl_doc=result.document)):
            page_numbers = set()
            bboxes = []
            labels = set()
            is_noise = False
            
            if hasattr(chunk.meta, 'doc_items'):
                for item in chunk.meta.doc_items:
                    label = str(item.label)
                    
                    # LOGIC: Filter out common "Noise" labels
                    if label in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]:
                        is_noise = True
                        break
                        
                    labels.add(label)
                    if hasattr(item, 'prov') and item.prov:
                        for p in item.prov:
                            page_numbers.add(p.page_no)
                            bboxes.append({"page": p.page_no, "bbox": [p.bbox.l, p.bbox.t, p.bbox.r, p.bbox.b]})
            
            if is_noise:
                continue

            chunks_data.append({
                "id": f"{pdf_path.stem}_{i:04d}",
                "text": chunk.text,
                "metadata": {
                    "source": pdf_path.name,
                    "headings": chunk.meta.headings if hasattr(chunk.meta, 'headings') else [],
                    "page_numbers": sorted(list(page_numbers)),
                    "labels": sorted(list(labels)),
                    "bboxes": bboxes,
                    "doc_title": result.document.name or pdf_path.stem
                }
            })

        json_path = job_dir / "chunks.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        duration = time.time() - start_time
        logger.info(f"Ingestion complete. {len(chunks_data)} chunks saved. Time: {duration:.2f}s")
        return str(json_path)
