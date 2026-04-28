import streamlit as st
import os
import time
import json
import tempfile
import fitz  # PyMuPDF for fast previews
from pathlib import Path
from PIL import Image

# Import your existing IngestionEngine logic
# Make sure your ingestion_engine.py is in the same directory
from ingestion_engine import IngestionEngine, resolve_page_range

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Ingestion & RAG Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Light Mode Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #1E1E1E; }
    .stSidebar { background-color: #F8F9FA; border-right: 1px solid #E0E0E0; }
    .main-header { font-size: 2rem; font-weight: 800; color: #0F172A; margin-bottom: 20px; }
    div.stButton > button:first-child { background-color: #2563EB; color: white; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_json" not in st.session_state:
    st.session_state.ingested_json = None

def main():
    st.markdown('<div class="main-header">📄 PDF Ingestion Engine</div>', unsafe_allow_html=True)

    # --- Sidebar: Upload and Config ---
    with st.sidebar:
        st.header("1. Data Setup")
        uploaded_file = st.file_uploader("Upload Source PDF", type=["pdf"])
        
        st.divider()
        st.header("2. Range Settings")
        col_start, col_end = st.columns(2)
        with col_start:
            skip_start = st.number_input("Skip Start", min_value=0, value=0, help="Pages to skip from beginning")
        with col_end:
            skip_end = st.number_input("Skip End", min_value=0, value=0, help="Pages to skip from end")
        
        output_dir = st.text_input("Output Storage", value="rag_storage")
        
        process_btn = st.button("🚀 Start Ingestion", use_container_width=True)

    # --- Main Body: 2-Column Layout ---
    preview_col, chat_col = st.columns([1, 1], gap="large")

    # Column 1: Document Preview
    with preview_col:
        st.subheader("Document Preview")
        if uploaded_file:
            # Save upload to a persistent temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Open with PyMuPDF for rendering
            doc = fitz.open(tmp_path)
            total_pages = len(doc)
            
            # Preview slider
            page_num = st.slider("Navigate Page", 1, total_pages, 1)
            
            # Render page to image
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            st.image(img, use_container_width=True, caption=f"Page {page_num} of {total_pages}")
            
            # Close doc to free memory
            doc.close()
        else:
            st.info("Upload a PDF in the sidebar to begin previewing.")

    # Column 2: Chat Environment
    with chat_col:
        st.subheader("Chat with your Data")
        
        # Display chat history
        chat_placeholder = st.container(height=500)
        with chat_placeholder:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_placeholder:
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    if st.session_state.ingested_json:
                        # Placeholder for RAG Retrieval Logic
                        # You would normally load st.session_state.ingested_json and search chunks
                        response = f"I've analyzed your document. Based on the {len(st.session_state.ingested_json)} chunks extracted, I am ready to answer."
                    else:
                        response = "Please run the ingestion process first so I can access the document content."
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    # --- Ingestion Execution ---
    if process_btn and uploaded_file:
        with st.spinner("Docling is converting and chunking..."):
            try:
                # 1. Resolve Range
                # We use your logic to convert 'skip' into 'page_range'
                page_range = resolve_page_range(tmp_path, skip_start, skip_end)
                
                # 2. Initialize Engine
                engine = IngestionEngine(output_root=output_dir)
                
                # 3. Process
                json_path = engine.process_pdf(tmp_path, page_range=page_range)
                
                # 4. Save result to session state
                with open(json_path, 'r', encoding='utf-8') as f:
                    st.session_state.ingested_json = json.load(f)
                
                st.success(f"Ingestion Complete! {len(st.session_state.ingested_json)} chunks saved.")
                st.balloons()
                
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

if __name__ == "__main__":
    main()