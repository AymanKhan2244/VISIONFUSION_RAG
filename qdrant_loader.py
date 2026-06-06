from qdrant_client import QdrantClient

from langchain_qdrant import (
    QdrantVectorStore
)

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME
)

from text_pipeline import (
    get_embedding_model
)


def load_vectorstore():

    embedding_model = (
        get_embedding_model()
    )

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )

    return vectorstore