import os
import time
import json
import logging
import traceback
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.labels import DocItemLabel
from docling.chunking import HierarchicalChunker

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_PATH = r"C:/docling_dist/models_cache"
os.environ["DOCLING_ARTIFACTS_PATH"] = BASE_PATH
os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─── User Input Collection ─────────────────────────────────────────────────────

def get_user_inputs_interactive() -> Tuple[str, int, int, str]:
    """
    Interactively prompt the user for all required inputs via the terminal.

    Returns
    -------
    pdf_path       : absolute/relative path to the PDF file
    skip_start     : number of pages to skip from the beginning
    skip_end       : number of pages to skip from the end
    output_root    : root directory where results will be saved
    """
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║            PDF Ingestion Engine  —  Input Setup          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ── PDF Path ──────────────────────────────────────────────────────────────
    while True:
        pdf_path = input("  📄  PDF file path  : ").strip().strip('"').strip("'")
        if not pdf_path:
            print("      ⚠  Path cannot be empty. Please try again.")
            continue
        if not Path(pdf_path).exists():
            print(f"      ⚠  File not found: {pdf_path!r}. Please check the path.")
            continue
        if Path(pdf_path).suffix.lower() != ".pdf":
            print("      ⚠  File does not appear to be a PDF. Continue anyway? (y/n) ", end="")
            if input().strip().lower() != "y":
                continue
        break

    # ── Pages to skip from START ───────────────────────────────────────────────
    while True:
        raw = input("  ⏭   Pages to skip from START (default 0): ").strip()
        if raw == "":
            skip_start = 0
            break
        if raw.isdigit():
            skip_start = int(raw)
            break
        print("      ⚠  Please enter a non-negative whole number.")

    # ── Pages to skip from END ─────────────────────────────────────────────────
    while True:
        raw = input("  ⏮   Pages to skip from END   (default 0): ").strip()
        if raw == "":
            skip_end = 0
            break
        if raw.isdigit():
            skip_end = int(raw)
            break
        print("      ⚠  Please enter a non-negative whole number.")

    # ── Output directory ───────────────────────────────────────────────────────
    raw = input("  📁  Output directory  (default: rag_storage): ").strip()
    output_root = raw if raw else "rag_storage"

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  Configuration Summary                              │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  PDF Path    : {pdf_path:<36} │")
    print(f"  │  Skip Start  : {skip_start:<36} │")
    print(f"  │  Skip End    : {skip_end:<36} │")
    print(f"  │  Output Dir  : {output_root:<36} │")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    confirm = input("  ▶   Proceed with ingestion? (y/n, default y): ").strip().lower()
    if confirm not in ("", "y", "yes"):
        print("  Aborted by user.")
        sys.exit(0)

    return pdf_path, skip_start, skip_end, output_root


def get_user_inputs_cli() -> Tuple[str, int, int, str]:
    """
    Parse all required inputs from command-line arguments.

    Returns
    -------
    Same tuple as get_user_inputs_interactive()
    """
    parser = argparse.ArgumentParser(
        description="PDF Ingestion Engine — convert, chunk, and store PDF content for RAG.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to ingest.",
    )
    parser.add_argument(
        "--skip-start",
        type=int,
        default=0,
        metavar="N",
        help="Number of pages to skip from the beginning of the PDF (default: 0).",
    )
    parser.add_argument(
        "--skip-end",
        type=int,
        default=0,
        metavar="N",
        help="Number of pages to skip from the end of the PDF (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rag_storage",
        metavar="DIR",
        help="Root directory where ingestion results will be saved (default: rag_storage).",
    )

    args = parser.parse_args()
    return args.pdf_path, args.skip_start, args.skip_end, args.output_dir


def resolve_page_range(
    pdf_path: str,
    skip_start: int,
    skip_end: int,
) -> Optional[Tuple[int, int]]:
    """
    Convert skip_start / skip_end values into a (first_page, last_page) tuple
    that Docling's page_range parameter understands (1-based, inclusive).

    Returns None when the full document should be processed (no skipping).
    Exits with an error message if the resulting range is invalid.
    """
    if skip_start == 0 and skip_end == 0:
        return None  # no filtering needed — let the original logic handle it

    # Determine total page count without a heavy conversion pass
    try:
        import pypdf  # lightweight; already installed as a docling dependency
        with open(pdf_path, "rb") as fh:
            total_pages = len(pypdf.PdfReader(fh).pages)
    except Exception:
        # Fallback: use pdfplumber or skip counting (the converter will clamp safely)
        total_pages = None

    first_page = 1 + skip_start
    last_page  = (total_pages - skip_end) if total_pages else sys.maxsize

    if first_page > last_page:
        print(
            f"\n  ✖  Invalid range: skipping {skip_start} from start and "
            f"{skip_end} from end leaves no pages to process."
        )
        sys.exit(1)

    logger.info(f"Page range resolved to: {first_page} – {last_page}")
    return (first_page, last_page)


# ─── Ingestion Engine (logic UNCHANGED) ───────────────────────────────────────

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

        # ── Final output summary ───────────────────────────────────────────────
        print()
        print("  ╔══════════════════════════════════════════════════════════╗")
        print("  ║                   Ingestion Complete ✔                   ║")
        print("  ╠══════════════════════════════════════════════════════════╣")
        print(f"  ║  Chunks saved   : {len(chunks_data):<38} ║")
        print(f"  ║  Time taken     : {duration:<35.2f}s ║")
        print(f"  ║  Output folder  : {str(job_dir):<38} ║")
        print(f"  ║  Chunks JSON    : {str(json_path):<38} ║")
        print(f"  ║  Markdown file  : {str(md_path):<38} ║")
        print("  ╚══════════════════════════════════════════════════════════╝")
        print()

        return str(json_path)


# ─── Entry Point ───────────────────────────────────────────────────────────────

def main():
    """
    Entry point.

    • Run with NO arguments  →  interactive prompts walk the user through setup.
    • Run with arguments     →  fully CLI-driven (useful for scripts / automation).

    Examples
    --------
    # Interactive mode
    python ingestion_engine.py

    # CLI mode
    python ingestion_engine.py report.pdf --skip-start 2 --skip-end 1 --output-dir my_output
    """
    # Decide input mode based on whether the user passed any args
    if len(sys.argv) > 1:
        pdf_path, skip_start, skip_end, output_root = get_user_inputs_cli()
    else:
        pdf_path, skip_start, skip_end, output_root = get_user_inputs_interactive()

    # Translate skip values → page_range understood by the existing engine logic
    page_range = resolve_page_range(pdf_path, skip_start, skip_end)

    # Run the engine (logic completely unchanged)
    engine = IngestionEngine(output_root=output_root)
    try:
        chunks_path = engine.process_pdf(pdf_path, page_range=page_range)
    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
