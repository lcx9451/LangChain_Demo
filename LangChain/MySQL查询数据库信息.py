from langchain_community.utilities import SQLDatabase
import os
import dotenv

dotenv.load_dotenv()  #加载当前目录下的 .env 文件

os.environ['OPENAI_API_KEY'] = os.getenv("DASHSCOPE_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("DASHSCOPE_BASE_URL")

# 方法1：直接传入 URI
# db = SQLDatabase.from_uri("mysql+pymysql://用户名:密码@主机:端口/数据库名")

# 方法2：分步配置
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
db = SQLDatabase.from_uri(uri)

result = db.run("SELECT `id`, `username`, `email`, `is_active`, `ai_quota`, `created_at`, `updated_at` FROM `users` WHERE `username` = 'test_user1' LIMIT 1")
print(result)










from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

# 初始化 LLM
llm = ChatOpenAI(model="qwen3-max", temperature=0)

# 创建 SQL 查询链
query_chain = create_sql_query_chain(llm, db)

# 执行查询
response = query_chain.invoke({"question": "查询用户test_user1的信息"})
print(response)














from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

# 初始化 LLM
llm = ChatOpenAI(model="qwen3-max", temperature=0)

# 创建 SQL Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

# 执行自然语言查询
response = agent.invoke("查询用户test_user1的邮箱是多少")
print(response)












from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# 1. 连接数据库
# db = SQLDatabase.from_uri("mysql+pymysql://root:password@localhost:3306/Chinook")

# 2. 初始化 LLM
llm = ChatOpenAI(model="qwen3-max", temperature=0)

# 3. 创建 SQL 查询链
write_query = create_sql_query_chain(llm, db)

# 4. 创建执行查询工具
execute_query = QuerySQLDataBaseTool(db=db)

# 5. 创建回答模板
answer_prompt = PromptTemplate.from_template("""
根据以下问题和 SQL 查询结果，用自然语言回答：
问题：{question}
SQL 查询：{query}
查询结果：{result}
回答：
""")

# 6. 组合完整链
answer_chain = answer_prompt | llm | StrOutputParser()

# 7. 完整流程
chain = (
    RunnablePassthrough.assign(query=write_query)
    .assign(result=lambda x: execute_query.invoke({"query": x["query"]}))
    | answer_chain
)

# 8. 执行
response = chain.invoke({"question": "找出用户的全部信息"})
print(response)