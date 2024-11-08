from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from pydantic import Field
from chatchat.server.utils import get_tool_config, build_logger
from .tools_registry import BaseToolOutput, regist_tool

logger = build_logger()


@regist_tool(title="SQL查询工具", description="Tool for querying a SQL database.")
def query_sql_data(query: str = Field(description="Execute a SQL query against the database and get back the result.")):
    """
    Execute a SQL query against the database and get back the result..
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    db_endpoint = get_tool_config("text2sql")["sqlalchemy_connect_str"]
    db = SQLDatabase.from_uri(db_endpoint)
    tool = QuerySQLDataBaseTool(db=db)
    logger.info(query)
    return BaseToolOutput(tool.run(query))
