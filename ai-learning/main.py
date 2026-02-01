from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
from worker import rag_worker

app = FastAPI(title="Ollama RAG Chatbot")


# -----------------------------
# Upload endpoint
# -----------------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    temp_name = f"temp_{uuid.uuid4()}_{file.filename}"

    with open(temp_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    rag_worker.add_document(temp_name)

    return {"message": "Document indexed successfully"}


# -----------------------------
# Ask endpoint
# -----------------------------
@app.get("/ask")
def ask_question(question: str):
    answer = rag_worker.ask(question)
    return {"answer": answer}
