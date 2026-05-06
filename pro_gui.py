import streamlit as st
import os
import time
import json
import tempfile
import fitz  # PyMuPDF for fast previews
from pathlib import Path
from PIL import Image

# Import the High-Performance Single-Pass Logic
from pro_ingestion_engine import SinglePassEngine, resolve_page_range

# --- Page Configuration ---
st.set_page_config(
    page_title="PRO PDF Ingestion",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #1E1E1E; }
    .main-header { font-size: 2.5rem; font-weight: 800; color: #1E40AF; margin-bottom: 10px; }
    .sub-text { font-size: 1.1rem; color: #64748B; margin-bottom: 30px; }
    div.stButton > button:first-child { background-color: #1E40AF; color: white; border-radius: 10px; height: 3em; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

if "ingested_json" not in st.session_state:
    st.session_state.ingested_json = None

def main():
    st.markdown('<div class="main-header">⚡ PRO Ingestion Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Single-Pass High-Performance Document Processing</div>', unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.header("📁 Upload & Configuration")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        
        st.divider()
        st.header("⚙️ Range Settings")
        skip_start = st.number_input("Skip Start (Pages)", min_value=0, value=0)
        skip_end = st.number_input("Skip End (Pages)", min_value=0, value=0)
        
        output_dir = st.text_input("Output Directory", value="pro_rag_results")
        
        st.divider()
        process_btn = st.button("🔥 RUN SINGLE-PASS", use_container_width=True)

    # --- Main Display ---
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📄 Document Preview")
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            doc = fitz.open(tmp_path)
            total_pages = len(doc)
            page_num = st.slider("Preview Page", 1, total_pages, 1)
            
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            st.image(img, use_container_width=True)
            doc.close()
        else:
            st.info("Upload a document to see the preview.")

    with col2:
        st.subheader("📊 Ingestion Status")
        if st.session_state.ingested_json:
            st.success(f"Success! {len(st.session_state.ingested_json)} chunks extracted.")
            st.json(st.session_state.ingested_json[:2]) # Preview first 2 chunks
        else:
            st.write("Ready for processing...")

    # --- Execution ---
    if process_btn and uploaded_file:
        with st.spinner("Executing high-performance single-pass conversion..."):
            try:
                # 1. Setup
                page_range = resolve_page_range(tmp_path, skip_start, skip_end)
                engine = SinglePassEngine(output_root=output_dir)
                
                # 2. Process (Whole file at once)
                json_path = engine.process_pdf(tmp_path, page_range=page_range)
                
                # 3. Load result
                with open(json_path, 'r', encoding='utf-8') as f:
                    st.session_state.ingested_json = json.load(f)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

if __name__ == "__main__":
    main()
