from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline


# Load lightweight AI model
qa_model = pipeline(
    "text-generation",
    model="gpt2"
)


def load_pdf(path):

    loader = PyPDFLoader(path)
    documents = loader.load()

    return documents


def create_vector_store(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        docs,
        embeddings
    )

    return db


def ask_question(db, question):

    docs = db.similarity_search(question, k=3)

    context = ""

    for doc in docs:
        context += doc.page_content + "\n"

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    result = qa_model(prompt, max_length=150)

    answer = result[0]["generated_text"]

    return answer
