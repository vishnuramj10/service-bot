import os
import json
import dotenv
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from utils import *
from waitress import serve
import time 
import requests

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
cors = CORS(app)
dotenv.load_dotenv()

AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

# Azure OpenAI settings
AZURE_OPENAI_KEY = "972c77672402466d8ba5345c4a048a3d"
AZURE_OPENAI_ENDPOINT = "https://azure-rag-openai.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

pdf_path = "document/POET_Everyday_Instructions.pdf"
vectordb = extract_text_and_create_embeddings(pdf_path, AZURE_SEARCH_KEY, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX)
keyword_image_map = create_keyword_mapping()

# Function to call the Azure OpenAI GPT-4 model
def invoke_azure_openai_model(prompt_text):
    try:
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_KEY
        }
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            "max_tokens": 2000
        }
        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error invoking Azure OpenAI model: {e}")
        return "Sorry, there was an error processing your request."

# Function to find the best matching keyword
def find_best_matching_keyword(user_query, keyword_image_map, threshold=0.5):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    user_query_embedding = model.encode([user_query])
    keyword_embeddings = model.encode(list(keyword_image_map.keys()))

    similarities = cosine_similarity(user_query_embedding, keyword_embeddings)[0]

    best_match_index = np.argmax(similarities)
    best_match_similarity = similarities[best_match_index]
    if best_match_similarity >= threshold:
        best_keyword = list(keyword_image_map.keys())[best_match_index]
        return best_keyword
    else:
        return None

# Function to process the chatbot query
def chatbot(query, vectordb, keyword_image_map):
    if not query.strip():
        return "Please ask a valid question.", []

    try:
        retrieved_docs = vectordb.similarity_search(query, k=4)
        relevant_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        best_keyword = find_best_matching_keyword(query, keyword_image_map)
        relevant_images = keyword_image_map.get(best_keyword, []) if best_keyword else []
        template = f"""
            You are a friendly, kind, and patient assistant for helpdesk. Act like a chatbot. 
            Your have to provide step-by-step instructions for user queries about using POET, based on the following content. 
            Do not add any information that is not present in the given content. If the query does not match the context, request the user to stay in context. 
            Format your response as follow if the query is related to the context:
            
                1. Start the response by "Surely I can help you. Here are the steps:". 
                2. Then, ALWAYS provide a bullet point list of all the steps.
                3. Number each step and provide clear, concise instructions.
                4. If there are sub-steps, use indented bullet points.
                5. Use exactly the same wording and formatting as in the original instructions.
                6. DO NOT skip any original instructions.

            Keep your answers to the point and don't include text from unrelated parts of the document.
            Do not include phrases like "based on the context given" or "based on this line." or any explanation
            <>

            {relevant_content}

            {query}
            Possible Answer:
        """
        
        prompt_template = PromptTemplate(template=template, input_variables=["relevant_content", "query"])
        prompt = prompt_template.format(relevant_content=relevant_content, query=query)

        response = invoke_azure_openai_model(prompt)
        return response, relevant_images
    except Exception as e:
        print(f"Error processing the chatbot query: {e}")
        return "Sorry, there was an error processing your request.", []

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/query', methods=['POST'])
def ask():
    try:
        query = request.form['message']
        response, image_paths = chatbot(query, vectordb, keyword_image_map)
    except Exception as e:
        print(f"Error handling request: {e}")
        response, image_paths = "Sorry, there was an error processing your request.", []

    response = response.replace('\n', '<br>') 
    return jsonify({'message': response, 'images': image_paths})

if __name__ == '__main__':
    app.run(debug=True)
