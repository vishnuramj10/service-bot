import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

def load_or_create_vector_db(pdf_path, db_dir="vectordb/"):
    # Check if the vector DB exists on disk
    if os.path.exists(os.path.join(db_dir, "index")):
        vectordb = Chroma(persist_directory=db_dir, embedding_function=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
        return vectordb

    # Otherwise, create and save the vector DB
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    # Split the pages into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)

    # Initialize embeddings
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create the vector database
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_dir,  # Specify where to save the DB
    )
    vectordb.persist()  

    return vectordb

# Function to create a mapping between keywords and images
def create_keyword_mapping():
    keyword_image_map = {
        "Every Morning": ["/static/extracted_images/page_1_img_1.png"],
        "Setting an active customer": ["/static/extracted_images/page_1_img_2.png", "/static/extracted_images/page_1_img_3.png"],
        "Sending an order": ["/static/extracted_images/page_4_img_1.png","/static/extracted_images/page_4_img_2.png"],
        "Verifying an order": ["/static/extracted_images/page_4_img_3.png","/static/extracted_images/page_4_img_4.png"]

    }
    return keyword_image_map