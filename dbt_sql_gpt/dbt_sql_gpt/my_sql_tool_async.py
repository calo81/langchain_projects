import os

from dotenv import load_dotenv
from langchain_core.tools import Tool
from pydantic.v1 import BaseModel

from dbt_sql_gpt.bigquery_client import BigQueryClient

load_dotenv()


class Query(BaseModel):
    query: str


class SqlSchema(BaseModel):
    sql_schema: str


class Dictionary(BaseModel):
    dictionary: dict


class MyTools:


    def run_query(self, query):
        try:
            project_id = os.getenv('GCP_PROJECT')
            credentials_path = os.getenv('GCP_CREDENTIALS')
            bigquery_client = BigQueryClient(project_id, credentials_path)
            result = bigquery_client.query(query)
            return result.to_dict()
        except Exception as e:
            print(e)
            return f"error: {str(e)}"


    def set_dataset(self, fed):
        return "Please introduce a SQL schema"

    def plot_data_func(self, dictionary):
        return f"Please do an HTML plot based on the data {dictionary}. Return the HTML only, no header, not markdown and no description at all"

    def run_query_tool(self):
        return Tool.from_function(
            name="run_query",
            description="run a SQL query, if an only if you already have a sql schema available to use. ALWAYS qualify the SQL with the sql schema",
            func=self.run_query,
            args_schema=Query
        )

    def set_dataset_tool(self):
        return Tool.from_function(
            name="set_dataset",
            description="sets a sql schema",
            func=self.set_dataset,
        )

    def plot_data_tool(self):
        return Tool.from_function(
            name="plot_data",
            description="creates all kind of html plots from the given data",
            func=self.plot_data_func,
            args_schema=Dictionary
        )