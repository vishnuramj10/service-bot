import os
import json
import dotenv
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from utils import *
import time 
import requests

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
cors = CORS(app)
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Azure OpenAI settings
AZURE_OPENAI_KEY = "972c77672402466d8ba5345c4a048a3d"
AZURE_OPENAI_ENDPOINT = "https://azure-rag-openai.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

pdf_path = "document/POET_Everyday_Instructions.pdf"
vectordb = extract_text_and_create_embeddings(pdf_path)
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
        print(response)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error invoking Azure OpenAI model: {e}")
        return "Sorry, there was an error processing your request."


def find_best_matching_keyword(user_query, response, keyword_image_map, threshold=0.378):
    model = OpenAIEmbeddings()

    # Encode the user query with the response
    user_query_embedding = np.array(model.embed_query(user_query + ' ' + response)).reshape(1, -1)  # Ensure it's 2D

    # Encode the keys in the keyword_image_map
    keyword_embeddings = np.array([model.embed_query(keyword) for keyword in keyword_image_map.keys()])

    # Compute cosine similarity between the user query and each keyword
    similarities = cosine_similarity(user_query_embedding, keyword_embeddings)

    # Find the index of the most similar keyword
    best_match_index = np.argmax(similarities)
    best_match_similarity = similarities[0][best_match_index]  # Get the similarity score

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
        #retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        #retrieved_docs = retriever.get_relevant_documents(query)  
        retrieved_docs = vectordb.similarity_search(query, k=4)  # Adjust 'k' to control the number of returned documents
        relevant_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        # best_keyword = find_best_matching_keyword(query, keyword_image_map)
        # relevant_images = keyword_image_map.get(best_keyword, []) if best_keyword else []
        template = f"""
            You are a friendly, kind, and patient AI assistant for helpdesk. Keep your conversations human-like.
            You have to provide step-by-step instructions for user queries about using POET, based on the following content. 
            Do not add any information that is not present in the given content. If the query does not match the context, request the user to stay in context.
            Please describe about what taks you can do in short when prompted.
            For relevant queries, start the response with "Here are the steps to perform....". You shouldn't say this for irrelevant queries. 
            Format your response as follow if the query is related to the context:
            
                1. ALWAYS provide a point list of all the steps.
                2. Number each step and provide clear, concise instructions.
                3. If there are sub-steps, use indented bullet points.
                4. Use exactly the same wording and formatting as in the original instructions.
                5. DO NOT skip any original instructions.
                6. If the user queries any basic conversational questions, respond in a detailed and understandable way and tell the user to ask about the relevant parts of the document.
                7. If the user query includes phrases which has "morning" in it, don't think of it as a conversational query and return the "every morning" instructions
                8. Sometimes the user can make mistakes while typing the query. So, please account for typos.

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
        best_keyword = find_best_matching_keyword(query, response, keyword_image_map)
        relevant_images = keyword_image_map.get(best_keyword, []) if best_keyword else []

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
    #serve(app, host="127.0.0.1", port = 8080)
