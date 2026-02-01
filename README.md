🚀 Local RAG Chatbot (FastAPI + Ollama + FAISS)
A lightweight Retrieval-Augmented Generation (RAG) chatbot that runs fully locally using:
⚡ FastAPI (API layer)
🧠 Ollama (LLM + embeddings)
📦 FAISS (vector database)
📄 PDF/TXT document ingestion

Upload documents → embed → store in FAISS → ask questions → get answers grounded in your data.

**Features**
✅ Upload PDF/TXT documents
✅ Automatic chunking + embeddings
✅ FAISS similarity search
✅ Context-aware answers using LLM
✅ Persistent storage (survives restart)
✅ Lightweight & fast
✅ Fully offline (Ollama local models)
✅ Simple 2-endpoint API

**Requirements**
Python 3.10+
Ollama installed
pip install fastapi uvicorn faiss-cpu ollama numpy pypdf
ollama pull mistral
ollama pull nomic-embed-text
ollama serve
