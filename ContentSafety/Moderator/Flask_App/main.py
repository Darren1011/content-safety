from flask import Flask, request, jsonify, render_template, redirect, url_for
from flasgger import Swagger
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import nltk

from azure.storage.blob import BlobServiceClient  
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # Import PorterStemmer
from nltk.tokenize import word_tokenize  # Import word_tokenize
from dotenv import load_dotenv
import re

nltk.download('stopwords')
# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
swagger = Swagger(app)

# Azure Blob Storage configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
BLOB_CONTAINER_NAME = os.getenv('BLOB_CONTAINER_NAME')

# Function to download the model and config from Azure Blob Storage
def download_model_from_blob():
    local_model_path = os.path.join(os.path.dirname(__file__), 'local_model_directory')
    os.makedirs(local_model_path, exist_ok=True)

    model_file_path = os.path.join(local_model_path, 'model.safetensors')
    config_file_path = os.path.join(local_model_path, 'config.json')

    if not os.path.exists(model_file_path) or not os.path.exists(config_file_path):
        print("Model files not found locally. Downloading from Azure Blob Storage...")

        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

        model_blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob='model.safetensors')
        with open(model_file_path, "wb") as model_file:
            model_file.write(model_blob_client.download_blob().readall())

        config_blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob='config.json')
        with open(config_file_path, "wb") as config_file:
            config_file.write(config_blob_client.download_blob().readall())

        print("Model and config files downloaded successfully.")
    else:
        print("Model files already exist locally. Skipping download.")

# Download the model when the application starts
download_model_from_blob()

# Load your fine-tuned BERT model and tokenizer from the local directory
local_model_path = os.path.join(os.path.dirname(__file__), 'local_model_directory')
model = BertForSequenceClassification.from_pretrained(local_model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Stopwords and preprocessing functions
stop_words = set(stopwords.words('english'))
stop_words.add("rt")  # adding rt to remove retweet in dataset

# Removing Emojis
def remove_entity(raw_text):
    entity_regex = r"&[^\s;]+;"
    text = re.sub(entity_regex, "", raw_text)
    return text

# Replacing user tags
def change_user(raw_text):
    regex = r"@([^ ]+)"
    text = re.sub(regex, "", raw_text)
    return text

# Removing URLs
def remove_url(raw_text):
    # Regex to detect URLs
    url_regex = r"(?i)\b(?:https?://|www\d{0,3}[.])[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|/))"
    # Replace full URL with 'http'
    text = re.sub(url_regex, 'http', raw_text)
    return text

# Removing Unnecessary Symbols
def remove_noise_symbols(raw_text):
    text = raw_text.replace('"', '').replace("'", '').replace("!", '').replace("`", '')
    text = text.replace("..", '').replace(".", '').replace(",", '').replace("#", '')
    text = text.replace(":", '').replace("?", '')
    return text

# Stemming
def stemming(raw_text):
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in raw_text.split()]
    return ' '.join(words)

# Removing stopwords
def remove_stopwords(raw_text):
    tokenize = word_tokenize(raw_text)
    text = [word for word in tokenize if not word.lower() in stop_words]
    text = ' '.join(text)
    return text

# Preprocessing pipeline
def preprocess(data):
    clean = []
    clean = [text.lower() for text in data]
    clean = [change_user(text) for text in clean]
    clean = [remove_entity(text) for text in clean]
    clean = [remove_url(text) for text in clean]
    clean = [remove_noise_symbols(text) for text in clean]
    clean = [stemming(text) for text in clean]
    clean = [remove_stopwords(text) for text in clean]

    return clean

# Text classification function
def classify_text(text):
    # Preprocess text
    text_list = [text]
    preprocessed_text = preprocess(text_list)[0]

    # Tokenize the text
    tokenized_input = tokenizer(preprocessed_text, return_tensors='pt')

    # Get model predictions
    output = model(**tokenized_input)
    logits = output.logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    if predicted_label == 0:
        predicted_label = "Appropriate"
    else:
        predicted_label = "Inappropriate"

    probability_class_0 = probabilities[0][0].item()
    probability_class_1 = probabilities[0][1].item()

    # Formulate response
    response = {
        "text": text,
        "predicted_class": predicted_label,
        "probabilities": {
            "appropriate": probability_class_0,
            "inappropriate": probability_class_1
        }
    }
    return response

@app.route('/')
def home():
    return redirect('/apidocs')

@app.route('/api/classify_text', methods=['POST'])
def api_classify_text():
    """
    API endpoint for classifying text as appropriate or inappropriate.
    ---
    parameters:
      - name: text
        in: formData
        type: string
        required: true
        description: The text to be classified.
    responses:
      200:
        description: The classification result
        schema:
          type: object
          properties:
            text:
              type: string
              description: The input text.
            predicted_class:
              type: string
              description: Predicted class (Appropriate or Inappropriate).
            probabilities:
              type: object
              properties:
                appropriate:
                  type: number
                  description: Probability of being appropriate.
                inappropriate:
                  type: number
                  description: Probability of being inappropriate.
    """
    text = request.form.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = classify_text(text)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)