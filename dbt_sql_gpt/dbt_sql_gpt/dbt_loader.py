import os
from typing import List

import yaml
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class DBTLoader(BaseLoader):
    def __init__(
        self,
        dbt_project_path: str
    ):
        """

        Args:
            dbt_project_path: Where is the dbt project
        """
        self.dbt_project_path = dbt_project_path

    def load(self) -> List[Document]:
        docs = []
        for yaml in self._get_yaml_files(self.dbt_project_path):
            for model in yaml['models']:
                for column in model['columns']:
                    prepared_text = self._prepare_text_to_index(model, column)
                    doc = Document(page_content=prepared_text,
                                   metadata={'model': model['name'], 'column': column['name'], 'table': model['name'],})
                    docs.append(doc)
        return docs

    def _prepare_text_to_index(self, model, column):
        return f"""column:  {column['name']} with description: {column.get('description', 'xx')}. belongs to model {model['name']}."""
    def _get_yaml_files(self, directory):
        yaml_files = []
        for filename in os.listdir(directory):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                yaml_files.append(self._load_yaml_file(directory + "/" + filename))
        return yaml_files

    def _load_yaml_file(self, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data

