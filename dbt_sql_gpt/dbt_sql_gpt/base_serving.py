import logging
import os
import tempfile
from enum import Enum

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dbt_sql_gpt.chat_model_start_handler import ChatModelStartHandler
from dbt_sql_gpt.custom_file_chat_message_history import CustomFileChatMessageHistory
from dbt_sql_gpt.dbt_loader import DBTLoader
from dbt_sql_gpt.my_sql_tool_async import MyTools


class LLMFlavor(Enum):
    OpenAI = 'OpenAI'
    Ollama = 'Ollama'


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
