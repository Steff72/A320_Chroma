
from typing import Any, List, Dict

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma

chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-4-1106-preview")
embeddings = OpenAIEmbeddings()
persist_directory = 'db'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True,
    )
    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm("What is DH?"))
