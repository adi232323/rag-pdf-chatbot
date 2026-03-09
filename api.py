from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from typing import List
import shutil
import os

from rag_pipeline import load_pdf, create_vector_store, ask_question

app = FastAPI()

vector_db = None


@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html") as f:
        return f.read()


@app.post("/upload")
async def upload_pdf(files: List[UploadFile] = File(...)):

    global vector_db

    os.makedirs("uploads", exist_ok=True)

    all_docs = []

    for file in files:

        file_path = f"uploads/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        docs = load_pdf(file_path)

        all_docs.extend(docs)

    vector_db = create_vector_store(all_docs)

    return {"message": f"{len(files)} PDF(s) uploaded successfully"}


@app.get("/ask")
def ask(question: str):

    global vector_db

    if vector_db is None:
        return {"answer": "Please upload a PDF first."}

    answer = ask_question(vector_db, question)

    return {"answer": answer}
