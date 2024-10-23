import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.azuresearch import AzureSearch

def extract_text_and_create_embeddings(pdf_path, AZURE_SEARCH_KEY, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    full_path = ''.join([dir_path, '/', pdf_path])
    print(f"Full pdf path: {full_path}")
    loader = PyPDFLoader(full_path)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint = AZURE_SEARCH_ENDPOINT,
        azure_search_key = AZURE_SEARCH_KEY,
        index_name = AZURE_SEARCH_INDEX,
        embedding_function=embeddings.embed_query,
        additional_search_client_options={"retry_total": 4}
    )
    vector_store.add_documents(documents=splits)
    
    return vector_store

# Function to create a mapping between keywords and images
def create_keyword_mapping():
    keyword_image_map = {
        "Every Morning": ["/static/extracted_images/page_1_img_1.png"],
        "Setting an active customer": ["/static/extracted_images/page_1_img_2.png", "/static/extracted_images/page_1_img_3.png"],
        "Sending an order": ["/static/extracted_images/page_4_img_1.png","/static/extracted_images/page_4_img_2.png"],
        "Verifying an order": ["/static/extracted_images/page_4_img_3.png","/static/extracted_images/page_4_img_4.png"]

    }
    return keyword_image_map