import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    SpacyTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

from retrieve_response import get_embedding_model

load_dotenv()

PDF_FILENAME = "uploaded.pdf"


def get_pdf_loader(loader_type):
    file_path = os.path.join(os.getcwd(), PDF_FILENAME)

    if loader_type == "PyPDF":
        return PyPDFLoader(file_path, extract_images=True)
    elif loader_type == "PyMuPDF":
        return PyMuPDFLoader(file_path, extract_images=True)
    elif loader_type == "Unstructured":
        return UnstructuredFileLoader(file_path)
    else:
        raise ValueError(f"Invalid loader type: {loader_type}")


def get_text_splitter(
    chunk_size, chunk_overlap, text_splitter, embedding, embedding_model
):
    if text_splitter == "RecursiveCharacter":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif text_splitter == "Character":
        return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif text_splitter == "Spacy":
        return SpacyTextSplitter(chunk_size=chunk_size)
    elif text_splitter == "SentenceTransformerToken":
        return SentenceTransformersTokenTextSplitter(tokens_per_chunk=chunk_size)
    elif text_splitter == "Semantic":
        return SemanticChunker(
            get_embedding_model(embedding, embedding_model)
        )  ### TODO Need to call embedding function to get the full embedding model
    else:
        raise ValueError(f"Invalid text splitter: {text_splitter}")


def load_docs(uploaded_files, document_loader):
    documents = []
    for file in uploaded_files:
        with open("uploaded.pdf", "wb") as f:
            f.write(file.getbuffer())
        loader = get_pdf_loader(document_loader)
        documents.extend(loader.load())

    # Add source name to metadata by removing the file extension
    for document in documents:
        document.metadata["source"] = os.path.splitext(
            os.path.basename(document.metadata["source"])
        )[0]
    return documents


def chunk_docs(
    documents, chunk_size, chunk_overlap, text_splitter, embedding_model, embedding
):
    text_splitter = get_text_splitter(
        chunk_size, chunk_overlap, text_splitter, embedding, embedding_model
    )
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents


def persist_docs(chunked_documents, embedding, embedding_model) -> Chroma:
    db = Chroma.from_documents(
        documents=chunked_documents,
        embedding=get_embedding_model(embedding, embedding_model),
        persist_directory="./chroma_db",
    )
    db = FAISS.from_documents(
        documents=chunked_documents,
        embedding=get_embedding_model(embedding, embedding_model),
    )
    db.save_local("faiss_index")
    return db
