# upload_handler.py
import os
from dotenv import load_dotenv
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create or connect to a Pinecone index
index_name = "business-documents"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Adjust based on your model's output dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Change to a supported region
        )
    )

index = pc.Index(index_name)

# Sentence Transformer model for vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')

# File upload directory
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

import json

def handle_file_upload(file):
    if file.filename == '':
        return {"success": False, "error": "No selected file"}

    if file and file.filename.endswith('.pdf'):
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)

        # Extract text from PDF
        text = extract_text_from_pdf(filename)

        # Vectorize the text
        vectors = model.encode(text)

        # Store vectors in Pinecone
        try:
            index.upsert([(file.filename, vectors.tolist())])
            
            # Save vectors and metadata locally
            local_data = {
                "filename": file.filename,
                "text": text,
                "vectors": vectors.tolist(),  # Convert numpy array to list for JSON serialization
            }

            # Save to a JSON file
            local_storage_path = os.path.join(UPLOAD_FOLDER, "local_storage.json")
            if os.path.exists(local_storage_path):
                with open(local_storage_path, "r") as f:
                    existing_data = json.load(f)
                existing_data.append(local_data)
            else:
                existing_data = [local_data]

            with open(local_storage_path, "w") as f:
                json.dump(existing_data, f, indent=4)

            return {"success": True, "message": "File uploaded, vectorized, and saved locally!"}
        except Exception as e:
            return {"success": False, "error": f"Failed to store vectors or save locally: {str(e)}"}

    return {"success": False, "error": "Invalid file type"}

def extract_text_from_pdf(filepath):
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text