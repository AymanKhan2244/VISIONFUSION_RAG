from vector_store import (
    store_text_documents,
    store_image_documents
)


def get_text_retriever(
    vectorstore,
    k=4
):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever




def retrieve_text(
    query,
    retriever
):

    results = retriever.invoke(
        query
    )

    return results




def retrieve_images(
    query,
    image_vectorstore,
    k=4
):

    results = image_vectorstore.similarity_search(
        query,
        k=k
    )

    return results




def build_context(
    text_results,
    image_results
):

    text_context = "\n".join(
        [
            doc.page_content
            for doc in text_results
        ]
    )

    image_context = "\n".join(
        [
            doc.page_content
            for doc in image_results
        ]
    )

    final_context = f"""
TEXT CONTEXT:

{text_context}


IMAGE CONTEXT:

{image_context}
"""

    return final_context