import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq


load_dotenv()


def get_llm(llm, llm_model, use_groq, temperature, max_output_tokens):
    if use_groq:
        return ChatGroq(model_name=llm_model, temperature=temperature, max_tokens=max_output_tokens)
    switch = {
        "OpenAI": get_openai_llm,
        "Meta": get_meta_llm,
        "Cohere": get_cohere_llm,
        "Mistral": get_mistral_llm,
        "Google": get_google_llm,
        "Anthropic": get_anthropic_llm,
        "Microsoft": get_microsoft_llm,
    }
    func = switch.get(llm)
    return func(llm_model, temperature, max_output_tokens)


def get_openai_llm(llm_model, temperature, max_output_tokens):
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature, max_tokens=max_output_tokens)


def get_meta_llm(llm_model, temperature, max_output_tokens):
    return ChatOllama(model=llm_model, temperature=temperature, num_predict=max_output_tokens)  ### TODO


def get_cohere_llm(llm_model, temperature, max_output_tokens):
    return ChatCohere(model=llm_model, temperature=temperature, max_tokens=max_output_tokens)  ### TODO


def get_mistral_llm(llm_model, temperature, max_output_tokens):
    return ChatOllama(model=llm_model, temperature=temperature, num_predict=max_output_tokens)  ### TODO


def get_google_llm(llm_model, temperature, max_output_tokens):
    if llm_model == "gemma:7b":
        return ChatOllama(model=llm_model, temperature=temperature, num_predict=max_output_tokens)
    return ChatGoogleGenerativeAI(model=llm_model, temperature=temperature, max_output_tokens=max_output_tokens)


def get_anthropic_llm(llm_model, temperature, max_output_tokens):
    return ChatAnthropic(model=llm_model, temperature=temperature, max_tokens=max_output_tokens)  ### TODO


def get_microsoft_llm(llm_model, temperature, max_output_tokens):
    return ChatOllama(model=llm_model, temperature=temperature, num_predict=max_output_tokens)  ### TODO


def get_embedding_model(embedding, embedding_model):
    switch = {
        "OpenAI": get_openai_embeddings,
        "Ollama": get_ollama_embeddings,
        "Cohere": get_cohere_embeddings,
        "Google": get_google_embeddings,
    }
    func = switch.get(embedding)
    return func(embedding_model)


def get_openai_embeddings(embedding_model):
    return OpenAIEmbeddings(model=embedding_model)


def get_ollama_embeddings(embedding_model):
    return OllamaEmbeddings(model=embedding_model)


def get_cohere_embeddings(embedding_model):
    return CohereEmbeddings(model=embedding_model)


def get_google_embeddings(embedding_model):
    return GoogleGenerativeAIEmbeddings(model=embedding_model)  ### TODO


def get_vector_db(vector_db, embedding_model):
    switch = {"Chroma": get_chroma_db, "Faiss": get_faiss_db}
    func = switch.get(vector_db)
    return func(embedding_model)


def get_chroma_db(embedding_model):
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model,
    )


def get_faiss_db(embedding_model):
    return FAISS.load_local("faiss_index", embedding_model)


def get_llm_response(query, embedding, embedding_model, vector_db, llm, llm_model, use_groq, temperature, max_output_tokens):
    template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always say "Thanks for asking!" at the end of the answer in a new line.
        
        Context : {context}
        
        Question : {question}
        
        Helpful Answer:
        
    """

    custom_rag_prompt = PromptTemplate.from_template(template)
    embedding_model = get_embedding_model(embedding, embedding_model)
    llm = get_llm(llm, llm_model, use_groq, temperature, max_output_tokens)
    vector_db = get_vector_db(vector_db, embedding_model)
    print(f"vector_db : {vector_db}")
    print(f"llm : {llm}")
    print(f"embedding_model : {embedding_model}")
    rag_chain = (
        {"context": vector_db.as_retriever(), "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )
    response = rag_chain.invoke(query)
    print(response)
    return response.content
