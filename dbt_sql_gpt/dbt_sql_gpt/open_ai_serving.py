from dotenv import load_dotenv
import logging

from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_core.document_loaders import BaseLoader
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

from dbt_sql_gpt.base_serving import LLMFlavor, BaseServing

# Configure logging
logging.basicConfig(level=logging.ERROR)

load_dotenv()


class MyGPTOpenAI(BaseServing):

    def __init__(self, static_context=""):
        super().__init__(static_context, LLMFlavor.OpenAI)
        self.llm = ChatOpenAI(model="gpt-4o", callbacks=[self.handler])

    def system_template(self):
        return """ {static_context}.
        You are an assistant to generate SQL queries
        
        You have 2 options:
        
        Option 1. If you don't have the sql schema from a previous message, you will just reply "Please introduce a dataset name in formar 'SQL schema: xx'"
         
        Option 2. if and Only If you already know the "sql schema" from a previous human message, Your answer will invoke the corresponding run_query tool with a SQL query. All tables in the query will be qualified with the sql schema. You will use the following pieces of context to answer the question at the end.
        
        context: {context}
        
        In any case, If you don't know the answer, ask me for more details, don't try to make up an answer.
        Always Use the context to find the correct column names. if you can't find the correct columns from the context, ask for more details.
        """

    def human_template(self):
        return """
        Question: {input}
        """

    async def run_llm_loop(self, loader: BaseLoader, input: str):

        chat_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self.system_template()),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(self.human_template()),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
        tools = [self.my_tools.run_query_tool(), self.my_tools.plot_data_tool()]

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

        async for message in agent_executor.astream(
            {'context': self.filtered_docs()(input), 'static_context': self.static_context,
             'input': input}):
            yield message
