# AI PDF Chatbot (RAG)

An AI-powered chatbot that allows users to upload PDF documents and ask questions about their content.

This project implements a Retrieval-Augmented Generation (RAG) pipeline using vector embeddings and a local language model.

## Features

- Upload multiple PDF documents
- Ask questions about the uploaded documents
- Vector search using ChromaDB
- AI-generated answers
- Simple ChatGPT-style web interface

## Tech Stack

Python  
FastAPI  
LangChain  
ChromaDB  
HuggingFace Transformers  
SentenceTransformers  

## How it Works

1. PDF documents are uploaded
2. Documents are split into smaller text chunks
3. Each chunk is converted into embeddings
4. Embeddings are stored in a vector database
5. User questions retrieve the most relevant chunks
6. A language model generates the final answer

## Installation

Clone the repository:

git clone https://github.com/adi232323/rag-pdf-chatbot.git

Install dependencies:

pip install -r requirements.txt

Run the server:

uvicorn api:app --host 0.0.0.0 --port 8000

Open the application:

http://localhost:8000

## Project Structure

rag-pdf-chatbot
│
├── api.py
├── rag_pipeline.py
├── index.html
├── requirements.txt
└── README.md
