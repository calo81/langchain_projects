import json
import logging
from enum import Enum
from typing import Any

from addict import Dict
from dotenv import load_dotenv
from langchain_core.document_loaders import BaseLoader
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_experimental.llms.ollama_functions import DEFAULT_RESPONSE_FUNCTION
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from dbt_sql_gpt.base_serving import BaseServing, LLMFlavor

# Configure logging
logging.basicConfig(level=logging.ERROR)

load_dotenv()


class OllamaFlavor(Enum):
    llama3_8b = "llama3:8b"
    llama3_70b = "llama3:70b"
    codestral = "codestral"


class MyGPTOllama(BaseServing):

    def __init__(self, ollama_flavor: OllamaFlavor = OllamaFlavor.llama3_8b):
        super().__init__(static_context="", llm_flavor=LLMFlavor.Ollama)
        self.ollama_flavor = ollama_flavor
        self.tools = [Dict(self._convert_to_ollama_tool(self.my_tools.run_query_tool())),
                      Dict(self._convert_to_ollama_tool(self.my_tools.plot_data_tool())),
                      DEFAULT_RESPONSE_FUNCTION]

        self.llm = OllamaFunctions(model=self.ollama_flavor.value, format="json", callbacks=[self.handler],
                                   tool_system_prompt_template=self.system_template(), keep_alive='120m')
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

    async def run_llm_loop(self, loader: BaseLoader, input: str):

        human_template = """
        Question: {input}. Remember to reply always using the provided tools. If the question is about plots, use the plot_data tool.
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
                {'context': self.filtered_docs()(input),
                 'input': input, "chat_history": self.memory.buffer_as_messages, 'tools': self.tools}).to_string())
        self.memory.chat_memory.add_user_message(input)
        self.memory.chat_memory.add_ai_message(message.content)

        if message.additional_kwargs.get('function_call', {}).get('name', None) == 'run_query':
            query = json.loads(message.additional_kwargs['function_call']['arguments'])['query']
            yield AIMessage(content=f"Running query {query}")
            result = self.my_tools.run_query(query)
            self.memory.chat_memory.add_ai_message(query)
            message = self.llm.invoke(
                chat_prompt.invoke(
                    {'context': self.filtered_docs()(input),
                     'input': f"For this message, reply back always in json using the tool: __conversational_response provided before with tool_input parameters response: 'The result of the query {query} is {result}' but nicely formatted",
                     "chat_history": self.memory.buffer_as_messages, 'tools': self.tools}).to_string())
            self.memory.chat_memory.add_ai_message(message.content)
            yield message
        elif message.additional_kwargs.get('function_call', {}).get('name', None) == 'plot_data':
            data = json.loads(message.additional_kwargs['function_call']['arguments'])['dictionary']
            yield AIMessage(content=f"Plotting Data for {data}")
            self.memory.chat_memory.add_ai_message(json.dumps(data))
            message = self.llm.invoke(
                chat_prompt.invoke(
                    {'context': self.filtered_docs()(input),
                     'input': f"For this message, reply back always in json using the tool: __conversational_response provided before with tool_input parameters response: an html plot of the data {data}",
                     "chat_history": self.memory.buffer_as_messages, 'tools': self.tools}).to_string())
            self.memory.chat_memory.add_ai_message(message.content)
            yield message
        else:
            yield message
