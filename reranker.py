from FlagEmbedding import FlagReranker


def load_reranker():

    reranker = FlagReranker(
    "BAAI/bge-reranker-base",
    use_fp16=False
)

    return reranker


def rerank_documents(
    query,
    documents,
    top_k=5
):

    reranker = load_reranker()

    pairs = [
        [query, doc.page_content]
        for doc in documents
    ]

    scores = reranker.compute_score(
        pairs
    )

    ranked_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc
        for doc, score
        in ranked_docs[:top_k]
    ]