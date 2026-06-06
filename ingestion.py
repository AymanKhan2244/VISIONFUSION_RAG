from text_pipeline import (
    load_pdf,
    chunk_documents,
    get_embedding_model
)

from image_pipeline import (
    extract_images,
    load_blip_model,
    generate_captions
)

from vector_store import (
    store_text_documents,
    store_image_documents
)


def ingest_pdf(pdf_path):

    print("Loading PDF...")

    docs = load_pdf(pdf_path)

    chunks = chunk_documents(docs)

    embeddings = get_embedding_model()

    print("Storing Text...")
    print(type(chunks))
    print(type(chunks[0]))
    text_store = store_text_documents(
        chunks,
        embeddings
    )

    print("Extracting Images...")

    image_paths = extract_images(
        pdf_path
    )

    processor, model = (
        load_blip_model()
    )

    captions = generate_captions(
        image_paths,
        processor,
        model
    )

    print("Storing Images...")

    image_store = store_image_documents(
        captions,
        image_paths,
        embeddings
    )

    return (
        text_store,
        image_store
    )