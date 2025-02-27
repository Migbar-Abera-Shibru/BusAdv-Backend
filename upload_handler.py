import os
from dotenv import load_dotenv
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_openai import AzureChatOpenAI  # Use AzureChatOpenAI instead of ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print(f"Pinecone API Key: {os.getenv('PINECONE_API_KEY')}")  # Debug log

# Sentence Transformer model for vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')

# Fixed index name
index_name = "business-documents"

# Create or connect to a Pinecone index (if it doesn't exist)
if index_name not in pc.list_indexes().names():
    print(f"Creating index: {index_name}")  # Debug log
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)
print(f"Connected to index: {index_name}")  # Debug log

# Initialize Azure OpenAI client
llm = AzureChatOpenAI(
    deployment_name=os.getenv("DEPLOYMENT_NAME"),  # Your Azure OpenAI deployment name
    openai_api_key=os.getenv("AZURE_API_KEY"),     # Your Azure OpenAI API key
    openai_api_version=os.getenv("OPENAI_API_VERSION"),  # Fetch API version from environment
    azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),      # Your Azure OpenAI endpoint
    temperature=0
)

def handle_file_upload(file, prompt):
    try:
        if file.filename == '':
            return {"success": False, "error": "No selected file"}

        if file and file.filename.endswith('.pdf'):
            print(f"Processing file: {file.filename}")  # Debug log

            # Extract text from PDF
            text = extract_text_from_pdf(file)
            print(f"Extracted text: {text[:100]}...")  # Debug log (first 100 characters)

            # Vectorize the text
            vectors = model.encode(text)
            print(f"Vectorized text: {vectors.tolist()[:5]}...")  # Debug log (first 5 vector values)

            # Use the document filename as the unique ID for the vector
            vector_id = file.filename.replace('.pdf', '').replace(' ', '_').lower()
            print(f"Vector ID: {vector_id}")  # Debug log

            # Truncate the metadata to stay within Pinecone's limit
            max_metadata_size = 40000  # 40 KB limit (leaving some buffer)
            truncated_text = text[:max_metadata_size]  # Store only the first 40,000 characters
            metadata = {"text": truncated_text}

            # Store vectors in Pinecone
            try:
                upsert_response = index.upsert([(vector_id, vectors.tolist(), metadata)])
                print(f"Pinecone upsert response: {upsert_response}")  # Debug log

                # If the prompt is a summarization request, generate a summary
                if "summarize" in prompt.lower():
                    summary = summarize_document(vector_id)
                    return {"success": True, "message": f"File uploaded and vectorized in Pinecone index '{index_name}'!", "summary": summary}
                else:
                    return {"success": True, "message": f"File uploaded and vectorized in Pinecone index '{index_name}'!"}
            except Exception as e:
                print(f"Error during Pinecone upsert: {e}")  # Debug log
                return {"success": False, "error": f"Failed to store vectors in Pinecone: {str(e)}"}

        return {"success": False, "error": "Invalid file type"}
    except Exception as e:
        print(f"Error in handle_file_upload: {e}")
        return {"success": False, "error": str(e)}

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def summarize_document(vector_id):
    try:
        # Retrieve the document text from Pinecone
        query_result = index.fetch(ids=[vector_id])

        if not query_result.vectors:
            print(f"No document found with ID: {vector_id}")  # Debug log
            return "No document found with the given ID."

        # Extract the document text from metadata
        document_text = query_result.vectors[vector_id].metadata.get("text", "")
        print(f"Retrieved document text: {document_text[:100]}...")  # Debug log (first 100 characters)

        # Summarize the document using LangChain and Azure OpenAI
        prompt = ChatPromptTemplate.from_template("Summarize the following document:\n\n{document}\n\nSummary:")
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"document": document_text})
        return summary
    except Exception as e:
        print(f"Error summarizing document: {e}")  # Debug log
        return f"Error summarizing document: {str(e)}"

def search_documents(query):
    try:
        # Vectorize the query
        query_vector = model.encode(query).tolist()

        # Query Pinecone
        query_result = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )

        if not query_result.matches:
            return "No matching documents found."

        # Extract relevant text from the results
        results = []
        for match in query_result.matches:
            results.append(match.metadata.get("text", ""))

        return results
    except Exception as e:
        return f"Error searching documents: {str(e)}"