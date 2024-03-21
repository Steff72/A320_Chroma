# https://gpt4all.io/index.html

from typing import Any, List, Dict

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All

chat = GPT4All(
        model="llm_model/nous-hermes-llama2-13b.Q4_0.gguf",
        verbose=True,
    )
embeddings = OpenAIEmbeddings()
persist_directory = 'db'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )
    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm("What is DH?"))
