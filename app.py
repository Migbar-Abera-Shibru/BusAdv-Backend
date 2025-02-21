from flask import Flask, request, jsonify
from openai import AzureOpenAI
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import SessionLocal, engine
from models import User, Advice  # Ensure models are imported
import os
from dotenv import load_dotenv
from flask_cors import CORS
import datetime
import re  # For regex-based intent detection and SQL generation
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Ensure database tables are created
from models import Base  # Import Base from models
Base.metadata.create_all(bind=engine)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
)

deployment_name = os.getenv("DEPLOYMENT_NAME")

# Initialize LangChain components
def init_database():
    db_uri = "mysql+pymysql://root:1234@127.0.0.1:3306/my_business_advisor"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
        
        <SCHEMA>{schema}</SCHEMA>
        
        Conversation History: {chat_history}
        
        Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
        Do not escape special characters like asterisks (*) in SQL queries.
        
        For example:
        Question: How many notifications are there for user Migbar?
        SQL Query: SELECT COUNT(*) FROM notifications WHERE UserID = (SELECT UserID FROM users WHERE Username = 'Migbar');
        Question: Show me all notifications for user Migbar.
        SQL Query: SELECT * FROM notifications WHERE UserID = (SELECT UserID FROM users WHERE Username = 'Migbar');
        
        Your turn:
        
        Question: {question}
        SQL Query:
        """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )



def validate_sql_query(sql_query):
    # Basic validation to ensure the query starts with SELECT, INSERT, UPDATE, or DELETE
    if not re.match(r"^\s*(SELECT|INSERT|UPDATE|DELETE)", sql_query, re.IGNORECASE):
        raise ValueError("Invalid SQL query. Only SELECT, INSERT, UPDATE, or DELETE queries are allowed.")
    
    # Check for escaped asterisks in SQL queries
    if "\\*" in sql_query:
        raise ValueError("Invalid SQL query. Do not escape asterisks in SQL queries.")
    
    return sql_query

def execute_sql_query(sql_query):
    db = SessionLocal()
    try:
        # Validate the SQL query
        sql_query = validate_sql_query(sql_query)
        
        # Execute the query
        result = db.execute(text(sql_query)).fetchall()
        return result
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return None
    finally:
        db.close()


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        # Debug: Print the generated SQL query
        generated_query = sql_chain.invoke({"question": user_query, "chat_history": chat_history})
        print(f"Generated SQL Query: {generated_query}")
        
        # Debug: Print the SQL response
        sql_response = db.run(generated_query)
        print(f"SQL Response: {sql_response}")
        
        response = chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
            "query": generated_query,
            "response": sql_response,
        })
        return response
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I couldn't process your request. Please try again."

        
# Function to determine if the prompt is database-related
def is_database_related(prompt):
    database_keywords = [
        "select", "insert", "update", "delete", "fetch", "search", "add", "remove", 
        "user", "users", "table", "database", "show", "list", "get", "find", "create", 
        "modify", "change", "record", "data", "row", "column"
    ]
    
    database_patterns = [
        r"\b(show|list|get|find)\b.*\b(users?|data|records?)\b",
        r"\b(how many|count)\b.*\b(users?|records?)\b",
        r"\b(add|insert)\b.*\b(users?|data|records?)\b",
        r"\b(update|modify|change)\b.*\b(users?|data|records?)\b",
        r"\b(delete|remove)\b.*\b(users?|data|records?)\b"
    ]
    
    if any(keyword in prompt.lower() for keyword in database_keywords):
        return True
    
    if any(re.search(pattern, prompt.lower()) for pattern in database_patterns):
        return True
    
    return False

# Function to process messages and interact with the database
def process_message(user_message):
    try:
        db = init_database()
        chat_history = []  # You can maintain chat history in session or database
        
        if is_database_related(user_message):
            print("Database-related prompt detected.")  # Debug log
            response_text = get_response(user_message, db, chat_history)
        else:
            print("General prompt detected.")  # Debug log
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "system", "content": "You are a helpful business advisor."},
                          {"role": "user", "content": user_message}],
                max_tokens=100,
                temperature=0.7,
            )
            response_text = response.choices[0].message.content.strip()

        return response_text

    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        response_text = process_message(user_message)
        return jsonify({"response": response_text})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)