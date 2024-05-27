from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from dbt_sql_gpt.dbt_loader import DBTLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dbt_sql_gpt.my_sql_tool import run_query_tool
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)

load_dotenv()


def filtered_docs(vectorstore):
    def inner(input):
        retrieved_docs = vectorstore.similarity_search_with_score(query=input, k=25)
        # find the more similar
        # filtered_docs = [doc[0] for doc in retrieved_docs if doc[1] < 0.4]
        return retrieved_docs
    return inner

def run_llm_loop(loader: BaseLoader):
    docs = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000, chunk_overlap=200, add_start_index=True
    # )
    # all_splits = text_splitter.split_documents(docs)
    #
    # len(all_splits)
    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6, "score_threshold": 0.8})
    #
    # retrieved_docs = retriever.invoke("What is the weather like?")


    llm = ChatOpenAI(model="gpt-4o")


    template = """Your answer will be a SQL query. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Also don't assume column names. Use the context to find their correct names. if you can't find the correct columns from the cntext, say you don't know.

    {context}

    Question: {input}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
            {"context": filtered_docs(vectorstore), "input": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
    )



    result = rag_chain.invoke("how much money has customer 'lucio sampaoli' spent and how many sessions he has had before conversion?")
    print(result)
    print("\n\n\n")

    result = rag_chain.invoke(
        "give me the first name of the top 5 biggest spenders that have at least 3 orders and have not used gift cards at all and have at least done 6 sessions")
    print(result)



if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    run_llm_loop(DBTLoader(f"{current_directory}/test_directory"))
