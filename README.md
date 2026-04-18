# Docling Offline RAG Suite 🦅

A professional-grade, fully offline Retrieval-Augmented Generation (RAG) system built with **Docling**, **ChromaDB**, and **Qwen2.5**. Designed specifically for high-accuracy parsing of technical PDFs, scientific papers, and complex tables.

## 🚀 Features

- **Layout-Aware Parsing**: Uses IBM's Docling to understand the hierarchy of your documents (Headers, Paragraphs, Tables, Formulas).
- **Formula Intelligence**: Automatically extracts and enriches LaTeX formulas using specialized vision models.
- **Hierarchical Chunking**: Unlike naive splitters, our system keeps context by linking every chunk to its parent heading.
- **100% Offline**: No internet calls. Your data stays on your machine.
- **Persistent Storage**: Efficiently query thousands of pages in milliseconds using ChromaDB.

## 📁 Project Structure

```text
├── src/
│   ├── ingestion_engine.py  # Docling parser & hierarchical chunker
│   ├── vector_engine.py     # Embedding management & ChromaDB
│   ├── generation_engine.py # Qwen2.5 LLM Reasoner
│   └── rag_manager.py       # CLI Orchestrator
├── models_cache/            # (Local only) Storage for AI model weights
├── rag_storage/             # (Local only) Your persistent vector database
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/docling-rag-offline.git
   cd docling-rag-offline
   ```

2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. **Download Model Weights**:
   This project requires weights for Parsing, Embedding, and Generation. Download them from our Hugging Face collection:
   👉 **[Link to Your Hugging Face Repo]**

   Place the files in the `models_cache/` directory following the structure provided in the HF repo.

## 📖 Usage

### 1. Ingest and Index a PDF
```bash
python src/rag_manager.py ingest "path/to/your/paper.pdf" --collection "My_Project"
```

### 2. Ask a Technical Question
```bash
python src/rag_manager.py query --collection "My_Project" --text "What is the formula for identity mapping?"
```

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Built with ❤️ for the Open Source AI Community.*
