from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME
)



def get_qdrant_client():

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    return client




def store_text_documents(
    chunks,
    embedding_model
):

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )

    return vectorstore




def store_image_documents(
    captions,
    image_paths,
    embedding_model
):

    image_docs = []

    for caption, image_path in zip(
        captions,
        image_paths
    ):

        image_docs.append(
            Document(
                page_content=caption,
                metadata={
                    "type": "image",
                    "image_path": image_path
                }
            )
        )

    vectorstore = QdrantVectorStore.from_documents(
        documents=image_docs,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )

    return vectorstore



def store_ocr_documents(
    ocr_docs,
    embedding_model
):

    docs = []

    for item in ocr_docs:

        docs.append(
            Document(
                page_content=item["text"],
                metadata={
                    "type": "ocr",
                    "image_path":
                    item["image_path"]
                }
            )
        )

    vectorstore = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )

    return vectorstore