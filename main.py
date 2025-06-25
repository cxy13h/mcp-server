import os, json, hashlib
from mcp.server.fastmcp import FastMCP
from datetime import datetime, date
from typing import Dict, Any, List, Annotated, Optional
from pydantic import Field
from fastmcp import FastMCP
from mcp.server.fastmcp.utilities.logging import get_logger

from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv
load_dotenv()
### Helpers ###

def tests_set_global(k, v):
    globals()[k] = v

### Database ###

def get_engine(readonly=True):
    connection_string = os.environ['DB_URL']
    # Get base engine options
    engine_options = {
        'isolation_level': 'AUTOCOMMIT',
        'execution_options': {'readonly': readonly}
    }
    
    db_engine_options = os.environ.get('DB_ENGINE_OPTIONS')
    
    if db_engine_options:
        try:
            custom_options = json.loads(db_engine_options)
            engine_options.update(custom_options)
            
        except json.JSONDecodeError:
            get_logger(__name__).warning("Invalid DB_ENGINE_OPTIONS JSON, ignoring")
        
    return create_engine(connection_string, **engine_options)

def get_db_info():
    engine = get_engine(readonly=True)
    with engine.connect():
        url = engine.url
        result = [
            f"Connected to {engine.dialect.name}",
            f"version {'.'.join(str(x) for x in engine.dialect.server_version_info)}",
            f"database {url.database}",
        ]

        if url.host:
            result.append(f"on {url.host}")

        if url.username:
            result.append(f"as user {url.username}")

        return " ".join(result) + "."

### Constants ###

VERSION = "2025.6.19.201831"
DB_INFO = get_db_info()
EXECUTE_QUERY_MAX_CHARS = int(os.environ.get('EXECUTE_QUERY_MAX_CHARS', 4000))
CLAUDE_LOCAL_FILES_PATH = os.environ.get('CLAUDE_LOCAL_FILES_PATH')

### MCP ###

mcp = FastMCP("TrySQL", host="0.0.0.0", json_response=True)
get_logger(__name__).info(f"Starting MCP Alchemy version {VERSION}")

@mcp.tool(
    name="all_table_names",
    description=f"返回数据库中所有表名，以','分隔。{DB_INFO}",
    tags={"database", "table", "list", "schema"},
)
def all_table_names() -> str:
    engine = get_engine()
    inspector = inspect(engine)
    return ", ".join(inspector.get_table_names())

@mcp.tool(
    name="filter_table_names",
    description=f"返回数据库中包含子字符串'q'的所有表名，以逗号分隔。{DB_INFO}",
    tags={"database", "table", "filter", "search"},
)
def filter_table_names(
    q: Annotated[
        str,
        Field(
            description="用于过滤表名的子字符串",
            type="string",
            example=["user", "order", "product"]
        )
    ]
) -> str:
    engine = get_engine()
    inspector = inspect(engine)
    return ", ".join(x for x in inspector.get_table_names() if q in x)

@mcp.tool(
    name="schema_definitions",
    description=f"返回给定表的模式和关系信息。{DB_INFO}",
    tags={"database", "schema", "structure", "columns", "relationships"},
)
def schema_definitions(
    table_names: Annotated[
        List[str],
        Field(
            description="要查询结构信息的表名列表",
            type="array",
            example=[["users", "orders"], ["products"], ["customers", "orders", "order_items"]]
        )
    ]
) -> str:
    def format(inspector, table_name):
        columns = inspector.get_columns(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        primary_keys = set(inspector.get_pk_constraint(table_name)["constrained_columns"])
        result = [f"{table_name}:"]

        # Process columns
        show_key_only = {"nullable", "autoincrement"}
        for column in columns:
            if "comment" in column:
                del column["comment"]
            name = column.pop("name")
            column_parts = (["primary key"] if name in primary_keys else []) + [str(
                column.pop("type"))] + [k if k in show_key_only else f"{k}={v}" for k, v in column.items() if v]
            result.append(f"    {name}: " + ", ".join(column_parts))

        # Process relationships
        if foreign_keys:
            result.extend(["", "    Relationships:"])
            for fk in foreign_keys:
                constrained_columns = ", ".join(fk['constrained_columns'])
                referred_table = fk['referred_table']
                referred_columns = ", ".join(fk['referred_columns'])
                result.append(f"      {constrained_columns} -> {referred_table}.{referred_columns}")

        return "\n".join(result)

    engine = get_engine()
    inspector = inspect(engine)
    return "\n".join(format(inspector, table_name) for table_name in table_names)

def execute_query_description():
    parts = [
        f"执行SQL查询并以可读格式返回结果。结果将在{EXECUTE_QUERY_MAX_CHARS}字符后截断。"
    ]
    if CLAUDE_LOCAL_FILES_PATH:
        parts.append("Claude Desktop可以通过URL获取完整结果集进行分析和制作工件。")
    parts.append(
        "重要提示：始终使用params参数进行查询参数替换（例如'WHERE id = :id'配合"
        "params={'id': 123}）以防止SQL注入。直接字符串拼接是严重的安全风险。")
    parts.append(DB_INFO)
    return " ".join(parts)

@mcp.tool(
    name="execute_query",
    description=execute_query_description(),
    tags={"database", "query", "sql", "select", "execute"},
)
def execute_query(
    query: Annotated[
        str,
        Field(
            description="要执行的SQL查询语句",
            type="string",
            example=[
                "SELECT * FROM users WHERE age > :age",
                "SELECT name, email FROM customers LIMIT 10",
                "SELECT o.*, c.name FROM orders o JOIN customers c ON o.customer_id = c.id"
            ]
        )
    ],
    params: Annotated[
        Dict[str, Any],
        Field(
            description="查询参数字典，用于安全的参数替换，防止SQL注入",
            type="object",
            default={},
            example=[{"age": 18}, {"status": "active", "limit": 100}]
        )
    ] = {}
) -> str:
    def format_value(val):
        """Format a value for display, handling None and datetime types"""
        if val is None:
            return "NULL"
        if isinstance(val, (datetime, date)):
            return val.isoformat()
        return str(val)

    def format_result(cursor_result):
        """Format rows in a clean vertical format"""
        result, full_results = [], []
        size, i, did_truncate = 0, 0, False

        i = 0
        while row := cursor_result.fetchone():
            i += 1
            if CLAUDE_LOCAL_FILES_PATH:
                full_results.append(row)
            if did_truncate:
                continue

            sub_result = []
            sub_result.append(f"{i}. row")
            for col, val in zip(cursor_result.keys(), row):
                sub_result.append(f"{col}: {format_value(val)}")

            sub_result.append("")

            size += sum(len(x) + 1 for x in sub_result)  # +1 is for line endings

            if size > EXECUTE_QUERY_MAX_CHARS:
                did_truncate = True
                if not CLAUDE_LOCAL_FILES_PATH:
                    break
            else:
                result.extend(sub_result)

        if i == 0:
            return ["No rows returned"], full_results
        elif did_truncate:
            if CLAUDE_LOCAL_FILES_PATH:
                result.append(f"Result: {i} rows (output truncated)")
            else:
                result.append(f"Result: showing first {i-1} rows (output truncated)")
            return result, full_results
        else:
            result.append(f"Result: {i} rows")
            return result, full_results

    def save_full_results(full_results):
        """Save complete result set for Claude if configured"""
        if not CLAUDE_LOCAL_FILES_PATH:
            return None

        def serialize_row(row):
            return [format_value(val) for val in row]

        data = [serialize_row(row) for row in full_results]
        file_hash = hashlib.sha256(json.dumps(data).encode()).hexdigest()
        file_name = f"{file_hash}.json"

        with open(os.path.join(CLAUDE_LOCAL_FILES_PATH, file_name), 'w') as f:
            json.dump(data, f)

        return (
            f"Full result set url: https://cdn.jsdelivr.net/pyodide/claude-local-files/{file_name}"
            " (format: [[row1_value1, row1_value2, ...], [row2_value1, row2_value2, ...], ...])"
            " (ALWAYS prefer fetching this url in artifacts instead of hardcoding the values if at all possible)")

    try:
        engine = get_engine(readonly=False)
        with engine.connect() as connection:
            cursor_result = connection.execute(text(query), params)

            if not cursor_result.returns_rows:
                return f"Success: {cursor_result.rowcount} rows affected"

            output, full_results = format_result(cursor_result)

            if full_results_message := save_full_results(full_results):
                output.append(full_results_message)

            return "\n".join(output)
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    main()