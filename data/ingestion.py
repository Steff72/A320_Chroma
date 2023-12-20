import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def ingest_docs() -> None:
    sources = ["data/fcom.pdf", "data/fctm.pdf"]
    documents = []

    for source in sources:
        loader = PyPDFLoader(source)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_split = text_splitter.split_documents(documents)
    print(f"loaded {len(doc_split) } documents")

    embeddings = OpenAIEmbeddings()
    persist_directory = "db"
    vectordb = Chroma.from_documents(
        documents=doc_split, embedding=embeddings, persist_directory=persist_directory
    )
    print("Docs added to Chroma DB")


if __name__ == "__main__":
    ingest_docs()
