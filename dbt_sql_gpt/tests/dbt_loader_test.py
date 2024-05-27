import pytest
from unittest.mock import patch, mock_open, MagicMock
import os
import yaml
from langchain_core.documents import Document
from dbt_sql_gpt.dbt_loader import DBTLoader


@pytest.fixture
def object_to_test():
    return DBTLoader('test_directory')

def test_load(object_to_test):
    docs = object_to_test.load()
    assert len(docs) == 1