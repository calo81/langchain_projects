from google.cloud import bigquery


class BigQueryClient:
    def __init__(self, project_id, credentials_path=None):
        """
        Initializes the BigQueryClient with the specified project ID and credentials.

        :param project_id: The ID of the GCP project.
        :param credentials_path: The path to the service account key file. If None, default credentials will be used.
        """
        if credentials_path:
            self.client = bigquery.Client.from_service_account_json(credentials_path, project=project_id)
        else:
            self.client = bigquery.Client(project=project_id)

    def query(self, query_string):
        """
        Executes a SQL query on BigQuery and returns the results as a DataFrame.

        :param query_string: The SQL query string to execute.
        :return: A pandas DataFrame containing the query results.
        """
        query_job = self.client.query(query_string)  # API request
        result = query_job.result()  # Wait for the job to complete
        df = result.to_dataframe()  # Convert the result to a pandas DataFrame
        return df
