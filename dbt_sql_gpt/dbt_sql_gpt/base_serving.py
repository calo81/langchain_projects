from enum import Enum


class LLMFlavor(Enum):
    OpenAI = 'OpenAI'
    Ollama = 'Ollama'


from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.chat_message_histories import FileChatMessageHistory, ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from dbt_sql_gpt.dbt_loader import DBTLoader
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dbt_sql_gpt.my_sql_tool_async import MyTools
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
import logging
from langchain.memory import ConversationBufferMemory
from dbt_sql_gpt.chat_model_start_handler import ChatModelStartHandler
from langchain_experimental.llms.ollama_functions import OllamaFunctions
import pprint
from langchain_core.document_loaders import BaseLoader
from dbt_sql_gpt.custom_file_chat_message_history import CustomFileChatMessageHistory
from enum import Enum
from langchain.agents.agent import Agent
from langchain.agents.structured_chat.base import create_structured_chat_agent
from langchain_experimental.llms.ollama_functions import convert_to_ollama_tool
from addict import Dict
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from dbt_sql_gpt.base_serving import LLMFlavor
import tempfile

# Configure logging
logging.basicConfig(level=logging.ERROR)

load_dotenv()


class BaseServing:

    def __init__(self, static_context="", llm_flavor=LLMFlavor.OpenAI):
        self.llm_flavor = llm_flavor
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        self.data_loader = DBTLoader(f"{current_directory}/test_directory")
        self.my_tools = MyTools()
        self.static_context = static_context
        self.memory = ConversationBufferMemory(
            chat_memory=CustomFileChatMessageHistory(tempfile.NamedTemporaryFile().name),
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output")
        self.handler = ChatModelStartHandler(self.memory)
        self.docs = self.data_loader.load()
        self.vectorstore = Chroma.from_documents(persist_directory="/tmp", documents=self.docs, embedding=OpenAIEmbeddings())

    def filtered_docs(self):
        def inner(input):
            retrieved_docs = self.vectorstore.similarity_search_with_score(query=input, k=25)
            docs = [doc[0].page_content for doc in retrieved_docs]
            return docs

        return inner

    async def run_llm_loop(self, loader: BaseLoader, input: str):
        docs = loader.load()
        vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())

        system_template = """ {static_context}.
        You are an assistant to generate SQL queries

        You have 2 options:

        Option 1. If you don't have the sql schema from a previous message, you will just reply "Please introduce a dataset name in formar 'SQL schema: xx'"

        Option 2. if and Only If you already know the "sql schema" from a previous human message, Your answer will invoke the corresponding run_query tool with a SQL query. All tables in the query will be qualified with the sql schema. You will use the following pieces of context to answer the question at the end.

        context: {context}

        In any case, If you don't know the answer, ask me for more details, don't try to make up an answer.
        Always Use the context to find the correct column names. if you can't find the correct columns from the context, ask for more details.
        """
        human_template = """
        Question: {input}
        """

        chat_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_template),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(human_template),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
        tools = [self.my_tools.run_query_tool()]

        agent = create_tool_calling_agent(
            llm=self.llm,
            prompt=chat_prompt,
            tools=tools
        )

        agent_executor = AgentExecutor(
            agent=agent,
            verbose=True,
            tools=tools,
            memory=self.memory,
            return_intermediate_steps=True
        )

        return agent_executor.astream(
            {'context': self.filtered_docs(vectorstore)(input), 'static_context': self.static_context,
             'input': input})
