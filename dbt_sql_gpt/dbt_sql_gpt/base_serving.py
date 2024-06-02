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
        self.vectorstore = Chroma.from_documents(persist_directory=tempfile.TemporaryDirectory().name, documents=self.docs, embedding=OpenAIEmbeddings())

    def filtered_docs(self):
        def inner(input):
            retrieved_docs = self.vectorstore.similarity_search_with_score(query=input, k=25)
            docs = set([doc[0].page_content for doc in retrieved_docs])
            return "\n".join(docs)

        return inner
