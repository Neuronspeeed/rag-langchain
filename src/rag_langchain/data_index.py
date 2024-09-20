import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import GitLoader
from rag_langchain.config import logger, OPENAI_API_KEY
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

def get_or_create_vectorstore():
    persist_directory = "./chroma_langchain_db"
    
    # Check if the vectorstore already exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        logger.info("Loading existing vectorstore...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        )
    else:
        logger.info("Creating new vectorstore...")
        loader = GitLoader(
            clone_url="https://github.com/langchain-ai/langchain",
            repo_path="./code_data/langchain_repo/",
            branch="master",
        )

        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=10000, chunk_overlap=100
        )

        docs = loader.load()
        docs = [doc for doc in docs if len(doc.page_content) < 50000]

        vectorstore = Chroma(
            collection_name="rag-chroma",
            embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
            persist_directory=persist_directory,
        )

        vectorstore.add_documents(documents=docs)
        vectorstore.persist()

    return vectorstore

vectorstore = get_or_create_vectorstore()
retriever = vectorstore.as_retriever(k=5)