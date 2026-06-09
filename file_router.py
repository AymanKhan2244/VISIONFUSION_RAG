import os

from document_ingestion import process_document
from image_ingestion import process_image


def process_file(file_path):

    ext = os.path.splitext(
        file_path
    )[1].lower()

    document_types = [
        ".pdf",
        ".docx",
        ".txt"
    ]

    image_types = [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp"
    ]

    if ext in document_types:

        return process_document(
            file_path
        )

    elif ext in image_types:

        return process_image(
            file_path
        )

    else:

        raise ValueError(
            f"Unsupported file type: {ext}"
        )