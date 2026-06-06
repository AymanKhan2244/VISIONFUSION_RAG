import os
from langchain_community .document_loaders import PyPDFLoader
from  langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


os.environ["HF_API_TOKEN"] = os.getenv("HF_API_TOKEN")


def load_pdf(pdf_path:str):
    """
        Load PDF and return raw documents
    """
    loader = PyPDFLoader(pdf_path)

    docs = loader.load()
    
   
    return docs
    
def chunk_documents(docs):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        docs
    )

    return chunks 


def get_embedding_model():
    """
        Load embedding model
    """
    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)
    return embeddings


