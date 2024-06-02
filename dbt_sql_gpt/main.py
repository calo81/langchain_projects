from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from dbt_sql_gpt.dbt_loader import DBTLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain.agents import AgentExecutor
from dbt_sql_gpt.my_sql_tool import MyTools
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
import logging
from langchain.memory import ConversationBufferMemory
from dbt_sql_gpt.chat_model_start_handler import ChatModelStartHandler
from langchain_core.document_loaders import BaseLoader
from dbt_sql_gpt.custom_file_chat_message_history import CustomFileChatMessageHistory
import logging
import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_core.document_loaders import BaseLoader
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from dbt_sql_gpt.chat_model_start_handler import ChatModelStartHandler
from dbt_sql_gpt.custom_file_chat_message_history import CustomFileChatMessageHistory
from dbt_sql_gpt.dbt_loader import DBTLoader
from dbt_sql_gpt.my_sql_tool import MyTools

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
        You can invoke 2 functions:
        Option 1. If you already know the "sql schema" from a previous message, Your answer will be a SQL query with all tables qualified with that sql schema. Use the following pieces of context to answer the question at the end.
        {context}
        
        Option 2. If you don't have the sql schema from a previous message, you will call the set_dataset function with empty parameter, and when you receive the response, then your next answer will follow Option 1.
        
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

        result = agent_executor.invoke({'context': self.filtered_docs(vectorstore)(input), 'static_context': self.static_context,  'input': input})
        return result

    def exit_shell(self, arg):
        print('Thank you for using my shell.')
        return True

    def run(self):
        while True:
            try:
                # input example: how much money has customer 'lucio sampaoli' spent and how many sessions he has had before conversion?
                user_input = self.session.prompt('ChatWH >> ')
                if user_input:
                    command, *args = user_input.split()
                    if command in self.commands:
                        if self.commands[command](" ".join(args)):
                            break
                    else:
                        self.run_llm_loop(self.data_loader, user_input)
            except (EOFError, KeyboardInterrupt):
                print('Exiting shell.')
                break


if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    # shell = MyGPT()
    shell = MyGPT(static_context="sql schema: cosas")
    shell.run()
