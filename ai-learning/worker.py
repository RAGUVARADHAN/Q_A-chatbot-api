# worker.py

import os
import faiss
import pickle
import numpy as np
import ollama
from pypdf import PdfReader

INDEX_PATH = "faiss_index/index.faiss"
META_PATH = "faiss_index/meta.pkl"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"


class RAGWorker:
    def __init__(self):
        os.makedirs("faiss_index", exist_ok=True)

        self.dimension = 768  # nomic embedding size
        self.index = None
        self.text_chunks = []

        self._load_index()

    # -----------------------------
    # Embedding
    # -----------------------------
    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for t in texts:
            res = ollama.embeddings(model=EMBED_MODEL, prompt=t)
            embeddings.append(res["embedding"])

        return np.array(embeddings).astype("float32")

    # -----------------------------
    # Chunking
    # -----------------------------
    def chunk_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks

    # -----------------------------
    # File Reading
    # -----------------------------
    def read_file(self, file_path):
        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            return "\n".join([p.extract_text() for p in reader.pages])
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

    # -----------------------------
    # Build / Save / Load FAISS
    # -----------------------------
    def _create_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)

    def _save_index(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.text_chunks, f)

    def _load_index(self):
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.text_chunks = pickle.load(f)
        else:
            self._create_index()

    # -----------------------------
    # Add Documents
    # -----------------------------
    def add_document(self, file_path):
        text = self.read_file(file_path)
        chunks = self.chunk_text(text)

        vectors = self.embed(chunks)

        self.index.add(vectors)
        self.text_chunks.extend(chunks)

        self._save_index()

    # -----------------------------
    # Query
    # -----------------------------
    def ask(self, question, k=1):
        q_vec = self.embed(question)

        D, I = self.index.search(q_vec, k)

        contexts = [self.text_chunks[i] for i in I[0]]

        context_text = "\n\n".join(contexts)

        prompt = f"""
Answer ONLY from the context.
If answer not found say "Not found in document".

Context:
{context_text}

Question:
{question}
"""

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]
        return context_text


rag_worker = RAGWorker()
