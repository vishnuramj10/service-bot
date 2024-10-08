import os
import json
import boto3
import dotenv
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from utils import *
import time 

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

dotenv.load_dotenv()

aws_access_key_id = os.getenv("AWS_ACCESS_KEY")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1") 

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)


pdf_path = "document/Copy of POET_Everyday_Instructions.pdf"
vectordb = load_or_create_vector_db(pdf_path)
print(vectordb)
print('hello')
keyword_image_map = create_keyword_mapping()

# Function to call the LLM model via Bedrock
def invoke_llama_model(prompt_text):
    try:
        model_id = "meta.llama3-70b-instruct-v1:0"
        payload = {
            "prompt": prompt_text,
            "max_gen_len": 2000,
        }

        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType='application/json',
            body=json.dumps(payload)
        )

        result = json.loads(response['body'].read().decode('utf-8'))
        return result['generation']
    except Exception as e:
        print(f"Error invoking LLM model: {e}")
        return "Sorry, there was an error processing your request."


def find_best_matching_keyword(user_query, keyword_image_map, threshold=0.5):
    try:
        model = OpenAIEmbeddings(model = 'text-embedding-3-large')

        keyword_embeddings = model.embed_documents(list(keyword_image_map.keys()))
        user_query_embedding = model.embed_documents([user_query])

        similarities = cosine_similarity(user_query_embedding, keyword_embeddings)[0]
        best_match_index = np.argmax(similarities)
        best_match_similarity = similarities[best_match_index]

        if best_match_similarity >= threshold:
            best_keyword = list(keyword_image_map.keys())[best_match_index]
            return best_keyword
        else:
            return None
    except Exception as e:
        print("Error:", e)
        return None

# Function to process the chatbot query
def chatbot(query, vectordb, keyword_image_map):
    if not query.strip():
        return "Please ask a valid question.", []

    try:
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        retrieved_docs = retriever.get_relevant_documents(query)  # Use the retriever here
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

        response = invoke_llama_model(prompt)
        print('hiuophfd')
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
        print(query)
        response, image_paths = chatbot(query, vectordb, keyword_image_map)
    except Exception as e:
        print(f"Error handling request: {e}")
        response, image_paths = "Sorry, there was an error processing your request.", []

    
    response = response.replace('\n', '<br>') 
    return jsonify({'message': response, 'images': image_paths})

if __name__ == '__main__':
    app.run(debug=True)
