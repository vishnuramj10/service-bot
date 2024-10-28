import os
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def extract_text_and_create_embeddings(pdf_path):
    # Load and split PDF
    dir_path = os.path.dirname(os.path.realpath(__file__))
    full_path = ''.join([dir_path, '/', pdf_path])
    print(f"Full pdf path: {full_path}")
    loader = PyPDFLoader(full_path)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Set up Chroma DB and add documents with embeddings
    vector_store = Chroma(collection_name="pdf_embeddings", embedding_function=embeddings)
    vector_store.add_documents(documents=splits)

    return vector_store

# Function to create a mapping between keywords and images
def create_keyword_mapping():
    keyword_image_map = {
        "Every Morning": ["/static/extracted_images/page_1_img_1.png"],
        "Setting an active customer": ["/static/extracted_images/page_1_img_2.png", "/static/extracted_images/page_1_img_3.png"],
        "Sending an order": ["/static/extracted_images/page_4_img_1.png","/static/extracted_images/page_4_img_3.png"],
        "Verifying an order": ["/static/extracted_images/page_4_img_2.png","/static/extracted_images/page_4_img_4.png"]

    }
    return keyword_image_map