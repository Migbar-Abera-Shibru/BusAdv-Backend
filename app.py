from flask import Flask, request, jsonify
import openai
import os
from dotenv import load_dotenv
from flask_cors import CORS  # Import CORS
from upload_handler import handle_file_upload

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
openai.api_key = os.getenv("AZURE_API_KEY")
openai.api_base = os.getenv("AZURE_API_ENDPOINT")
openai.api_type = "azure"
openai.api_version = "2024-08-01-preview"
deployment_name = os.getenv("DEPLOYMENT_NAME")

@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,  # Azure-specific: Use deployment name here
            messages=[
                {"role": "system", "content": "You are a helpful business advisor."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=100,
            temperature=0.7,
        )
        bot_message = response.choices[0].message["content"].strip()
        return jsonify({"choices": [{"text": bot_message}]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400

    file = request.files['file']
    result = handle_file_upload(file)

    if result["success"]:
        return jsonify(result), 200
    else:
        return jsonify(result), 400

if __name__ == "__main__":
    app.run(debug=True)