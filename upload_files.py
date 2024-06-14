import time
import streamlit as st
from retrieve_response import get_llm_response
from load_chunk_embed_persist import (
    load_docs,
    chunk_docs,
    persist_docs,
)

from dotenv import load_dotenv

load_dotenv()

st.title("Chat with Multiple PDFs")

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Configure RAG Parameters and Upload Documents")

    st.text("Select the appropriate parameters for your RAG application:")

    document_loader_options = ["PyPDF", "PyMuPDF", "Unstructured"]
    document_loader = st.selectbox("Choose an Document Loader", document_loader_options)

    text_splitter_options = [
        "Character",
        "RecursiveCharacter",
        "Spacy",
        "SentenceTransformerToken",
        "Semantic",
    ]
    text_splitter = st.selectbox("Choose a Text Splitter", text_splitter_options)

    col1, col2 = st.columns([1, 1])

    with col1:
        chunk_size = st.number_input(min_value=0, label="Chunk Size", value=1000)

    with col2:
        # Setting the max value of chunk_overlap based on chunk_size
        chunk_overlap = st.number_input(
            min_value=0,
            label="Chunk Overlap",
            value=int(chunk_size * 0.1),
        )

        # Display a warning if chunk_overlap is not less than chunk_size
        if chunk_overlap >= chunk_size:
            st.warning("Chunk Overlap should be less than Chunk Length!")

    embedding_options = ["OpenAI", "Ollama", "Cohere", "Google"]
    embedding = st.selectbox("Choose an Embedding option", embedding_options)

    ### TODO Put all the correct embedding models for the embedding providers
    embedding_model_options = {
        "OpenAI": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
        "Ollama": ["mxbai-embed-large", "nomic-embed-text", "all-minilm"],
        "Cohere": [
            "embed-english-v3.0",
            "embed-english-light-v3.0",
            "embed-multilingual-v3.0",
        ],
        "Google": [
            "text-embedding-004",
            "text-multilingual-embedding-002",
            "textembedding-gecko@003",
            "textembedding-gecko-multilingual@001",
        ],
    }
    embedding_model = st.selectbox(
        "Choose an Embedding Model", embedding_model_options[embedding]
    )

    vector_db_options = ["Chroma", "Faiss", "Pinecone"]
    vector_db = st.selectbox("Choose a Vector DB", vector_db_options)

    
    use_groq = st.checkbox("Use Groq", value=False)

    if use_groq:
        llm_options = ["Meta","Mistral", "Google"]
        llm = st.selectbox("Choose an LLM", llm_options)
        llm_model_options = {
            "Meta": ["llama3-70b-8192", "llama3-8b-8192"],
            "Mistral": ["mixtral-8x7b-32768"],
            "Google": ["gemma-7b-it"],
        }
        llm_model = st.selectbox("Choose an LLM Model", llm_model_options[llm])
    else:
        llm_options = [
            "OpenAI",
            "Meta",
            "Cohere",
            "Mistral",
            "Google",
            "Anthropic",
            "Microsoft",
        ]
        llm = st.selectbox("Choose an LLM", llm_options)

        llm_model_options = {
            "OpenAI": ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"],
            "Meta": ["llama2", "llama3:8B"],
            "Cohere": ["command-r", "command-r-plus", "c4ai-aya-23"],
            "Mistral": ["mistral-7b", "mixtral-8x7b"],
            "Google": ["gemini-1.5-flash", "gemini-1.0-pro", "gemma:7b"],
            "Anthropic": ["claude-3-haiku-20240307", "claude-2.1"],
            "Microsoft": ["phi3:medium", "phi3:mini"],
        }
        llm_model = st.selectbox("Choose an LLM Model", llm_model_options[llm])

    col3, col4 = st.columns([1, 1])
    
    with col3:
        temperature = st.slider(min_value=0.0, max_value=1.0, label="Temperature", value=0.4)
    
    with col4:
        max_output_tokens = st.slider(min_value=0, label="Max Output Tokens", value=1024, max_value=8192)

    uploaded_files = st.file_uploader(
        "Upload PDF files here", type=["pdf"], accept_multiple_files=True
    )

    placeholder = st.empty()

    if st.button("Submit"):
        # pass
        if uploaded_files is not None:
            with st.spinner("Processing"):
                placeholder.write(f"Loading Documents using {document_loader}")
                documents = load_docs(uploaded_files, document_loader)
                placeholder.write(
                    f"Chunking Documents using {text_splitter} with chunk size as {chunk_size} and chunk overlap as {chunk_overlap}"
                )
                chunked_documents = chunk_docs(
                    documents,
                    chunk_size,
                    chunk_overlap,
                    text_splitter,
                    embedding,
                    embedding_model,
                )  ### TODO Need to pass embedding
                placeholder.write(
                    f"Persisting Documents in {vector_db} with embedding model as {embedding_model} of {embedding}"
                )
                persist_docs(chunked_documents)
                placeholder.write("Documents persisted successfully")
            placeholder.write("Now you can chat with your uploaded PDF")
            time.sleep(5)
            placeholder.empty()
        else:
            st.write("Please upload a PDF file to see its content.")

query = st.text_input("Ask me anything about your PDFs")
submit = st.button("Ask")

if submit:
    with st.spinner("Processing"):
        st.write(
            get_llm_response(
                query,
                embedding,
                embedding_model,
                vector_db,  ### TODO Need to work on passing the embedding model
                llm,
                llm_model,
                use_groq,
                temperature,
                max_output_tokens,
            )
        )
        placeholder.write(f"This response is served by {llm_model} model of {llm}")
