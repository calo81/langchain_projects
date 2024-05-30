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
from langchain_experimental.llms.ollama_functions import DEFAULT_RESPONSE_FUNCTION
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
from langchain_core.pydantic_v1 import BaseModel
from typing import Any
import json

# Configure logging
logging.basicConfig(level=logging.ERROR)

load_dotenv()


class MyGPTOllama:

    def __init__(self):
        self.llm_flavor = LLMFlavor.Ollama
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        self.data_loader = DBTLoader(f"{current_directory}/test_directory")
        self.my_tools = MyTools()
        self.memory = ConversationBufferMemory(
            chat_memory=CustomFileChatMessageHistory("../memory.json"),
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output")
        handler = ChatModelStartHandler(self.memory)
        self.tools = [Dict(self._convert_to_ollama_tool(self.my_tools.run_query_tool())),
                      DEFAULT_RESPONSE_FUNCTION]

        self.llm = OllamaFunctions(model="llama3:70b", format="json", callbacks=[handler],
                                   tool_system_prompt_template=self.system_template())
        self.llm = self.llm.bind_tools(tools=self.tools)

    def _is_pydantic_class(self, obj: Any) -> bool:
        return isinstance(obj, type) and (
                issubclass(obj, BaseModel) or BaseModel in obj.__bases__
        )

    def _convert_to_ollama_tool(self, tool: Any) -> Dict:
        """Convert a tool to an Ollama tool."""

        if self._is_pydantic_class(tool.__class__):
            schema = tool.__dict__["args_schema"].schema()
            definition = {"name": tool.name, "properties": schema["properties"], "description": tool.description}
            if "required" in schema:
                definition["required"] = schema["required"]

            return definition
        raise ValueError(
            f"Cannot convert {tool} to an Ollama tool. {tool} needs to be a Pydantic model."
        )

    def system_template(self):
        return """
        You are an assistant to generate SQL queries
        
        You have access to the following tools:

       {tools}

       You must always select one of the above tools and respond with only a JSON object matching the following schema:

       {{
          "tool": <name of the selected tool>,
          "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
       }}
       """

    def system_template_2(self):
        return """
        You will invoke these tools as follows 
        
        Option 1. If you don't have the sql schema from a previous message, you will invoke __conversational_response tool"
         
        Option 2. if and Only If you already know the "sql schema" from a previous human message, You will invoke the  run_query tool with a SQL query. All tables in the query will be qualified with the sql schema. 
        
        You will always use the following pieces of context to answer the question at the end.
        
        context: {context}
        
        Always Use the context to find the correct column names. if you can't find the correct columns from the context, ask for more details invoking the __conversational_response tool.
        """

    def filtered_docs(self, vectorstore):
        def inner(input):
            retrieved_docs = vectorstore.similarity_search_with_score(query=input, k=25)
            docs = [doc[0].page_content for doc in retrieved_docs]
            return docs

        return inner

    def run_llm_loop(self, loader: BaseLoader, input: str):
        docs = loader.load()
        vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())

        human_template = """
        Question: {input}
        """

        chat_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_template()),
                SystemMessagePromptTemplate.from_template(self.system_template_2()),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(human_template),
            ]
        )
        message = self.llm.invoke(
            chat_prompt.invoke(
                {'context': self.filtered_docs(vectorstore)(input),
                 'input': input, "chat_history": self.memory.buffer_as_messages, 'tools': self.tools}).to_string())

        self.memory.chat_memory.add_user_message(input)
        self.memory.chat_memory.add_ai_message(message.content)

        if message.additional_kwargs.get('function_call', {}).get('name', None) == 'run_query':
            query = json.loads(message.additional_kwargs['function_call']['arguments'])['query']
            result = self.my_tools.run_query(query)
            self.memory.chat_memory.add_ai_message(query)
            message = self.llm.invoke(
                chat_prompt.invoke(
                    {'context': self.filtered_docs(vectorstore)(input),
                     'input': f"For this message, reply me back using tool __conversational_response with the following verbatim 'The result of the query {query} is {result}' but nicely formatted", "chat_history": self.memory.buffer_as_messages, 'tools': self.tools}).to_string())
            self.memory.chat_memory.add_ai_message(message.content)

        async def list_to_aiter(lst):
            for item in lst:
                yield item

        return list_to_aiter([message])
