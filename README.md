# 🧠 Local RAG Assistant – Private Document Q&A with Ollama

![Demo](docs/demo.gif)

A fully local, privacy‑focused AI assistant that answers questions **only from your own documents** (PDF, Word, TXT). Built with **Ollama**, **Streamlit**, and **sentence‑transformers**. No cloud, no API keys, no recurring costs.

## ✨ Features

- 🔒 **100% offline** – your data never leaves your computer
- 📄 Supports **PDF, DOCX, TXT** files
- 🧠 **Hybrid search** (vector + BM25) for accurate retrieval
- 🎯 **Exact answer extraction** for dates, numbers, fees
- 💬 Clean **chat interface** (Streamlit)
- 🖥️ Works on **Windows, macOS, Linux** (with Ollama)

## 🚀 Quick Start

### Prerequisites
- [Ollama](https://ollama.com) installed and running
- A model pulled, e.g.  
  ```bash
  ollama pull llama3.2:3b
