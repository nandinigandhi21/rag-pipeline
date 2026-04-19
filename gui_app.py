import os
import sys
import time
import json
import streamlit as st
import pypdfium2 as pdfium
from pathlib import Path
from PIL import Image
from io import BytesIO

# Add src to path for imports
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from ingestion_engine import IngestionEngine
from vector_engine import VectorEngine
from generation_engine import GenerationEngine

# --- CACHED ENGINES ---
@st.cache_resource
def get_vector_engine():
    return VectorEngine()

@st.cache_resource
def get_generation_engine():
    # Only load LLM when actually chatting to save RAM
    return GenerationEngine()

# --- UTILITIES ---
def clear_memory():
    # Force streamlit to drop cached resources
    st.cache_resource.clear()
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.success("Memory Cleared!")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Eagle Eye RAG 🦅",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME MANAGEMENT (Custom CSS) ---
def apply_custom_theme(mode):
    if mode == "Purplish-Dark":
        st.markdown("""
            <style>
            /* Global Background and Text */
            .stApp {
                background-color: #0d1117;
                color: #f0f6fc;
            }
            
            /* Sidebar Styling */
            [data-testid="stSidebar"] {
                background-color: #161b22;
                border-right: 1px solid #30363d;
            }

            /* Headers and Labels */
            h1, h2, h3, p, span, label, .stMarkdown {
                color: #f0f6fc !important;
            }

            /* Buttons */
            .stButton>button {
                background-color: #238636;
                color: white !important;
                border-radius: 6px;
                border: 1px solid rgba(240,246,252,0.1);
                padding: 0.5rem 1rem;
                font-weight: 600;
            }
            .stButton>button:hover {
                background-color: #2ea043;
                border-color: #8b949e;
            }

            /* Input Fields */
            .stTextInput>div>div>input, .stNumberInput>div>div>input, .stTextArea>div>div>textarea {
                background-color: #0d1117 !important;
                color: #f0f6fc !important;
                border: 1px solid #30363d !important;
                border-radius: 6px;
            }
            
            /* Chat Messages */
            .chat-message.user {
                background-color: #1f6feb;
                color: #ffffff !important;
                padding: 1.2rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                border-bottom-right-radius: 2px;
            }
            .chat-message.bot {
                background-color: #21262d;
                color: #f0f6fc !important;
                padding: 1.2rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                border: 1px solid #30363d;
                border-bottom-left-radius: 2px;
            }

            /* Expander Styling */
            .p-expander-content {
                background-color: #161b22 !important;
                border: 1px solid #30363d !important;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp { background-color: #f8f9fa; }
            .stButton>button { border-radius: 8px; }
            </style>
        """, unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/DS4SD/docling/main/docs/assets/docling_logo.png", width=150)
    st.title("Control Center")
    
    theme_mode = st.radio("UI Theme", ["Purplish-Dark", "Light Mode"])
    apply_custom_theme(theme_mode)
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Technical PDF", type="pdf")
    
    if uploaded_file:
        # Save temp file for processing
        temp_path = Path("temp_upload.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("PDF Uploaded!")
        
        # Page Range Selection
        pdf = pdfium.PdfDocument(str(temp_path))
        max_pages = len(pdf)
        
        st.subheader("Parsing Options")
        start_pg = st.number_input("Start Parsing at Page", min_value=1, max_value=max_pages, value=1)
        end_pg = st.number_input("End Parsing at Page", min_value=start_pg, max_value=max_pages, value=max_pages)
        
        collection_name = st.text_input("Collection Name", value=uploaded_file.name.split('.')[0][:20])
        
        if st.button("🚀 Process & Index Document"):
            with st.spinner("Analyzing Layout & Extracting Knowledge..."):
                try:
                    # Modular Loading: Load ingestor locally and kill it after
                    ingestor = IngestionEngine()
                    chunks_path = ingestor.process_pdf(str(temp_path), page_range=(start_pg, end_pg))
                    del ingestor
                    
                    # Force Garbage Collection to free RAM for LLM
                    import gc
                    gc.collect()
                    
                    vector_engine = get_vector_engine()
                    vector_engine.populate_from_json(chunks_path, collection_name)
                    
                    st.session_state.pdf_processed = True
                    st.session_state.current_collection = collection_name
                    st.success("Index Ready!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    st.divider()
    if st.button("🧹 Clear System Memory"):
        clear_memory()

# --- MAIN INTERFACE ---
st.title("Eagle Eye RAG 🦅")
st.caption("High-Accuracy Technical Document Reasoning")

tab1, tab2 = st.tabs(["💬 Technical Chat", "🖼️ PDF Preview"])

with tab2:
    if uploaded_file:
        st.subheader(f"Document Preview: {uploaded_file.name}")
        pdf = pdfium.PdfDocument("temp_upload.pdf")
        
        # Show thumbnails for page range selection
        cols = st.columns(4)
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=0.5)
            pil_image = bitmap.to_pil()
            
            with cols[i % 4]:
                st.image(pil_image, caption=f"Page {i+1}", use_container_width=True)
                if i+1 < start_pg or i+1 > end_pg:
                    st.caption("❌ Will be skipped")
                else:
                    st.caption("✅ Will be parsed")
    else:
        st.info("Upload a PDF in the sidebar to see the preview.")

with tab1:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a technical question..."):
        if not st.session_state.pdf_processed:
            st.warning("Please process the document in the sidebar first.")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        v_engine = get_vector_engine()
                        results = v_engine.query(st.session_state.current_collection, prompt)
                        context_chunks = results['documents'][0]
                        
                        gen_engine = get_generation_engine()
                        answer = gen_engine.generate_answer(prompt, context_chunks)
                        
                        st.markdown(answer)
                        
                        # Show Sources in Expander
                        with st.expander("View Context Sources"):
                            for i, meta in enumerate(results['metadatas'][0]):
                                st.write(f"**Source {i+1} (Page {json.loads(meta['page_numbers'])}):**")
                                st.code(context_chunks[i], language="markdown")
                                
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error: {e}")
