import os
import sys
import time
import json
import streamlit as st
import pypdfium2 as pdfium
from pathlib import Path
from PIL import Image
from io import BytesIO

# =============================================================================
# SMART PERMANENT OFFLINE FIX
# =============================================================================
try:
    from transformers.configuration_utils import PretrainedConfig
    _orig_from_dict_func = PretrainedConfig.from_dict.__func__
    @classmethod
    def _smart_patched_from_dict(cls, config_dict, **kwargs):
        config = _orig_from_dict_func(cls, config_dict, **kwargs)
        if isinstance(config, dict) and not hasattr(config, 'to_dict'):
            class ConfigWrapper(dict):
                def to_dict(self): return self
            return ConfigWrapper(config)
        return config
    PretrainedConfig.from_dict = _smart_patched_from_dict
except Exception:
    pass
# =============================================================================

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
    return GenerationEngine()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Document Querying System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CLEAN LIGHT THEME ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #dee2e6; }
    .stButton>button { border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# --- SIDEBAR ---
with st.sidebar:
    st.title("Control Center")
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Technical PDF", type="pdf")
    
    if uploaded_file:
        temp_path = Path("temp_upload.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("PDF Uploaded!")
        pdf = pdfium.PdfDocument(str(temp_path))
        max_pages = len(pdf)
        
        st.subheader("Parsing Options")
        page_range = st.slider("Select Page Range", 1, max_pages, (1, max_pages))
        start_pg, end_pg = page_range
        
        collection_name = st.text_input("Collection Name", value=uploaded_file.name.split('.')[0][:20])
        
        if st.button("🚀 Process & Index"):
            with st.spinner("Analyzing Layout..."):
                try:
                    ingestor = IngestionEngine()
                    chunks_path = ingestor.process_pdf(str(temp_path), page_range=(start_pg, end_pg))
                    vector_engine = get_vector_engine()
                    vector_engine.populate_from_json(chunks_path, collection_name)
                    st.session_state.pdf_processed = True
                    st.session_state.current_collection = collection_name
                    st.success("Ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- MAIN INTERFACE ---
st.title("Document Querying System")
st.caption("High-Accuracy Technical Document Reasoning (Fully Offline)")

tab1, tab2 = st.tabs(["💬 Chat", "📄 Preview"])

with tab2:
    if uploaded_file:
        st.subheader(f"Preview: {uploaded_file.name}")
        pdf = pdfium.PdfDocument("temp_upload.pdf")
        cols = st.columns(4)
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=0.5)
            pil_image = bitmap.to_pil()
            with cols[i % 4]:
                st.image(pil_image, caption=f"Page {i+1}", width='stretch')
                status = "✅" if (i+1 >= start_pg and i+1 <= end_pg) else "❌"
                st.caption(f"{status} Page {i+1}")
    else:
        st.info("Upload a PDF to see preview.")

with tab1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the document..."):
        if not st.session_state.pdf_processed:
            st.warning("Please process the document first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    v_engine = get_vector_engine()
                    results = v_engine.query(st.session_state.current_collection, prompt)
                    context_chunks = results['documents'][0]
                    gen_engine = get_generation_engine()
                    answer = gen_engine.generate_answer(prompt, context_chunks)
                    st.markdown(answer)
                    with st.expander("Sources"):
                        for i, meta in enumerate(results['metadatas'][0]):
                            st.write(f"Source {i+1} (Page {json.loads(meta['page_numbers'])}):")
                            st.code(context_chunks[i], language="markdown")
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
