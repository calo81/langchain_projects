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
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dbt_sql_gpt.my_sql_tool import MyTools
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
import logging
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from dbt_sql_gpt.chat_model_start_handler import ChatModelStartHandler
from langchain_experimental.llms.ollama_functions import OllamaFunctions
import pprint

# Configure logging
logging.basicConfig(level=logging.ERROR)

load_dotenv()


class MyGPT:

    def __init__(self, static_context=""):
        self.session = PromptSession(history=InMemoryHistory())
        self.commands = {
            'exit': self.exit_shell,
        }
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        self.data_loader = DBTLoader(f"{current_directory}/test_directory")
        self.my_tools = MyTools(self.session)
        self.static_context = static_context
        self.memory = ConversationBufferMemory(
            chat_memory=FileChatMessageHistory("memory.json"),
            memory_key="chat_history",
            return_messages=True,
            input_key="input")
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

    async def run_llm_loop(self, loader: BaseLoader, input: str):
        docs = loader.load()
        vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())



        system_template = """ {static_context}.
        You can invoke 2 functions:
        Option 1. If you already know the "sql schema" from a previous message, Your answer will be a SQL query with all tables qualified with that sql schema. Use the following pieces of context to answer the question at the end.
        {context}
        
        Option 2. If you don't know the sql schema, you will call to the set_dataset function with 'xx' as parameter, and when you receive the response, then your next answer will follow Option 1.
        
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

        agent = OpenAIFunctionsAgent(
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

        import pprint

        chunks = []

        async for chunk in agent_executor.astream(
                {'context': self.filtered_docs(vectorstore)(input), 'static_context': self.static_context,  'input': input}
        ):
            chunks.append(chunk)
            print("------")
            pprint.pprint(chunk, depth=1)

        # result = agent_executor({'context': self.filtered_docs(vectorstore)(input), 'static_context': self.static_context,  'input': input})
        # return result
    def exit_shell(self, arg):
        print('Thank you for using my shell.')
        return True

    async def run(self):
        while True:
            try:
                # input example: how much money has customer 'lucio sampaoli' spent and how many sessions he has had before conversion?
                user_input = await self.session.prompt_async('>> ')
                if user_input:
                    command, *args = user_input.split()
                    if command in self.commands:
                        if self.commands[command](" ".join(args)):
                            break
                    else:
                        await self.run_llm_loop(self.data_loader, user_input)
            except (EOFError, KeyboardInterrupt):
                print('Exiting shell.')
                break

import asyncio
if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    shell = MyGPT()
    # shell = MyGPT(static_context="sql schema: otrova")
    asyncio.run(shell.run())
