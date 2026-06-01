from langchain_community.document_loaders import PyPDFLoader
from  langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import  Chroma ,FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv()



user = input("upload the file: ")



loader = PyPDFLoader(user)

document = loader.load()


texts = [doc.page_content for doc in document]

print(document[0])
text_splitter = RecursiveCharacterTextSplitter(
    chunk_overlap=50,
    chunk_size=500
)


chunks = text_splitter.split_documents(document)


print(len(chunks))