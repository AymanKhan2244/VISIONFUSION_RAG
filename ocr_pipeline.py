import easyocr


def load_ocr():

    reader = easyocr.Reader(
        ['en'],
        gpu=False
    )

    return reader


def extract_ocr_text(
    image_paths,
    reader
):

    ocr_documents = []

    for image_path in image_paths:

        result = reader.readtext(
            image_path,
            detail=0
        )

        text = " ".join(result)

        ocr_documents.append(
            {
                "image_path": image_path,
                "text": text
            }
        )

        print(
            f"OCR Extracted from {image_path}"
        )

    return ocr_documents