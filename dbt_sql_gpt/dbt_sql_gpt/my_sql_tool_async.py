import os
from google.cloud import bigquery
from dbt_sql_gpt.bigquery_client import BigQueryClient
import pandas as pd
from dotenv import load_dotenv
from pydantic.v1 import BaseModel
from langchain_core.tools import Tool

load_dotenv()


class Query(BaseModel):
    query: str


class SqlSchema(BaseModel):
    sql_schema: str


class MyTools:


    def run_query(self, query):
        try:
            project_id = os.getenv('GCP_PROJECT')
            credentials_path = os.getenv('GCP_CREDENTIALS')
            bigquery_client = BigQueryClient(project_id, credentials_path)
            result = bigquery_client.query(query)
            return result
        except Exception as e:
            print(e)
            return "no result"


    def set_dataset(self, fed):
        return "Please introduce a SQL schema"

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
