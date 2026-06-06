from qdrant_loader import (
    load_vectorstore
)

from reranker import (
    rerank_documents
)

from llm import (
    generate_answer
)

print(
    "Loading Vector Database..."
)

vectorstore = (
    load_vectorstore()
)

retriever = (
    vectorstore.as_retriever(
        search_kwargs={
            "k": 15
        }
    )
)

print(
    "VisionFusion RAG Ready!"
)

while True:

    query = input(
        "\nAsk Question: "
    )

    if query.lower() == "exit":
        break

    

    results = retriever.invoke(
        query
    )

    
    reranked_results = results[:5]

    

    context = "\n\n".join(
        [
            doc.page_content
            for doc in results
        ]
    )

   

    answer = generate_answer(
        query,
        context
    )

    print("\n")
    print("=" * 50)
    print("ANSWER")
    print("=" * 50)

    print(answer)

    print("\n")
    print("=" * 50)
    print("SOURCES")
    print("=" * 50)

    for i, doc in enumerate(
        results,
        start=1
    ):

        print(
            f"\nSource {i}"
        )

        print(
            doc.metadata
        )