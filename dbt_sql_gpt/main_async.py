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

# Configure logging
logging.basicConfig(level=logging.ERROR)

load_dotenv()


class MyGPT:

    def __init__(self, static_context="", chunks=[]):
        self.chunks = chunks
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        self.data_loader = DBTLoader(f"{current_directory}/test_directory")
        self.my_tools = MyTools()
        self.static_context = static_context
        self.memory = ConversationBufferMemory(
            chat_memory=CustomFileChatMessageHistory("memory.json"),
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output")
        handler = ChatModelStartHandler(self.memory)
        self.llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
        # self.llm = Ollama(model="llama3:70b")
        # self.llm = OllamaFunctions(model="llama3:70b", format="json")
        # self.llm.bind_tools([self.my_tools.run_query_tool(), self.my_tools.set_dataset_tool()])

    def filtered_docs(self, vectorstore):
        def inner(input):
            retrieved_docs = vectorstore.similarity_search_with_score(query=input, k=25)
            # find the more similar
            # filtered_docs = [doc[0] for doc in retrieved_docs if doc[1] < 0.4]
            return retrieved_docs

        return inner

    def run_llm_loop(self, loader: BaseLoader, input: str):
        docs = loader.load()
        vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())

        system_template = """ {static_context}.
        You can invoke 2 tools:
        Option 1. Only If you already know the "sql schema" from a previous human message, Your answer will invoke the corresponding run_query tool with a SQL query. All tables in the query will be qualified with the sql schema. You will use the following pieces of context to answer the question at the end.
        
        context:{context}
        
        Option 2. If you don't have the sql schema from a previous message, you will call the set_dataset tool with empty parameter, and when you receive the response, you will get the SQL Schema from it and then your next answer will follow Option 1.
        
        In any case, If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always Use the context to find the correct column names. if you can't find the correct columns from the context, say you don't know.
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

        tools = [self.my_tools.run_query_tool(), self.my_tools.set_dataset_tool()]

        agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=chat_prompt,
            tools=tools
        )

        agent_executor = AgentExecutor(
            agent=agent,
            verbose=True,
            tools=tools,
            memory=self.memory,
            return_intermediate_steps=True,
        )

        return agent_executor.astream(
                {'context': self.filtered_docs(vectorstore)(input), 'static_context': self.static_context,
                 'input': input})



if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    # shell = MyGPT()
    shell = MyGPT(static_context="sql schema: cosas")
    shell.run()
