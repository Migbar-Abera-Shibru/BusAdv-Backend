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
from upload_handler import handle_file_upload, summarize_document, search_documents

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
        Ensure that the query checks for existing data to avoid duplicates.
        
        For example:
        Question: How many rows are in the advice table?
        SQL Query: SELECT COUNT(*) FROM advice;
        Question: Add advice for user Meron.
        SQL Query: INSERT INTO advice (UserID, Context, AdviceText, CreatedAt) SELECT UserID, 'print results', 'print your results', NOW() FROM users WHERE Username = 'Meron' AND NOT EXISTS (SELECT 1 FROM advice WHERE UserID = (SELECT UserID FROM users WHERE Username = 'Meron') AND Context = 'print results' AND AdviceText = 'print your results');
        
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

# Function to determine if the prompt is vector database-related
def is_vector_database_related(prompt):
    vector_keywords = [
        "vector", "pinecone", "search", "document", "summarize", "upload", "file", "pdf"
    ]
    
    if any(keyword in prompt.lower() for keyword in vector_keywords):
        return True
    
    return False

# Function to process messages and interact with the database
def process_message(user_message):
    try:
        db = init_database()
        chat_history = []  # You can maintain chat history in session or database
        
        # Debug: Log the user message
        print(f"User message: {user_message}")

        if is_database_related(user_message):
            print("MySQL database-related prompt detected.")  # Debug log
            response_text = get_response(user_message, db, chat_history)
        elif is_vector_database_related(user_message):
            print("Vector database-related prompt detected.")  # Debug log
            response_text = handle_vector_database_prompt(user_message)
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
        print(f"Error in process_message: {e}")  # Debug log
        return f"Error: {str(e)}"
        
def handle_vector_database_prompt(prompt):
    if "summarize" in prompt.lower():
        # Extract the document name or ID from the prompt
        # For example: "Summarize the document with ID 123"
        # This is a placeholder; you'll need to implement the logic to extract the document ID or name.
        document_id = "example_document_id"
        summary = summarize_document(document_id)
        return summary
    elif "search" in prompt.lower():
        # Extract the search query from the prompt
        # For example: "Search for documents about business plans"
        query = prompt.replace("search", "").strip()
        results = search_documents(query)
        return results
    else:
        return "Sorry, I couldn't process your request. Please try again."
    
@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        print(f"Received message: {user_message}")  # Debug log
        response_text = process_message(user_message)
        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Error in /api/chat: {e}")  # Debug log
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    prompt = request.form.get("prompt", "")  # Extract the prompt from the form data

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Pass both the file and prompt to handle_file_upload
        result = handle_file_upload(file, prompt)
        return jsonify(result)
    except Exception as e:
        print(f"Error in upload endpoint: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.json
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    summary = summarize_document(filename)
    return jsonify({"summary": summary})

@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    results = search_documents(query)
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)