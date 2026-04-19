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

