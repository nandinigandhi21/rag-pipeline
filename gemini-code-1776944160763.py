import json
import pandas as pd
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
# Update this path to where your ingestion_engine.py saved the chunks.json
CHUNKS_JSON_PATH = r"rag_storage/your_job_folder/chunks.json" 
OUTPUT_CSV = "ingestion_audit_report.csv"

def generate_audit_report(json_path: str):
    """
    Parses chunks.json to create a line-by-line audit of headings, 
    labels, and sizes to evaluate Docling's performance.
    """
    path_obj = Path(json_path)
    if not path_obj.exists():
        print(f"❌ Error: File not found at {json_path}")
        return

    print(f"📂 Loading data from: {path_obj.name}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Extraction Phase (Line-by-Line)
    raw_entries = []
    for chunk in data:
        meta = chunk.get("metadata", {})
        headings = meta.get("headings", [])
        labels = ", ".join(meta.get("labels", []))
        chunk_size = len(chunk.get("text", ""))
        # Convert list of pages to a readable string
        pages = ", ".join(map(str, meta.get("page_numbers", [])))

        if not headings:
            raw_entries.append({
                "Heading Text": "No Heading (Body Text)",
                "Level": 0,
                "Labels": labels,
                "Size (Chars)": chunk_size,
                "Page(s)": pages
            })
        else:
            # Create a unique row for every level of heading associated with this chunk
            for level, text in enumerate(headings, start=1):
                raw_entries.append({
                    "Heading Text": text,
                    "Level": level,
                    "Labels": labels,
                    "Size (Chars)": chunk_size,
                    "Page(s)": pages
                })

    df = pd.DataFrame(raw_entries)

    # 2. Summary & Aggregation Phase
    # We group by heading and label to see how many chunks are living under each title
    summary_table = df.groupby(["Heading Text", "Level", "Labels", "Page(s)"]).agg({
        "Size (Chars)": ["count", "mean", "sum"]
    }).reset_index()

    # Flatten and rename columns for a professional audit look
    summary_table.columns = [
        "Heading Text", 
        "Heading Level", 
        "Content Labels", 
        "Page Numbers", 
        "Chunk Count", 
        "Avg Chunk Size", 
        "Total Section Size"
    ]

    # 3. Sorting for Document Flow
    # Ensures the table follows the actual sequence of the PDF
    summary_table = summary_table.sort_values(by=["Page Numbers", "Heading Level"])

    # 4. Save and Display
    summary_table.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*60)
    print("                INGESTION AUDIT COMPLETE                ")
    print("="*60)
    print(f"Total Unique Headings Processed: {len(summary_table['Heading Text'].unique())}")
    print(f"Audit Report Saved To: {OUTPUT_CSV}")
    print("="*60)
    
    # Display top 15 rows for a quick terminal preview
    print("\nPreview of Audit Results:")
    print(summary_table.head(15).to_string(index=False))

if __name__ == "__main__":
    generate_audit_report(CHUNKS_JSON_PATH)