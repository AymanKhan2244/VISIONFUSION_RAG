from image_pipeline import (
    extract_images
)

from ocr_pipeline import (
    load_ocr,
    extract_ocr_text
)

pdf_path = "Attention.pdf"

image_paths = extract_images(
    pdf_path
)

reader = load_ocr()

ocr_docs = extract_ocr_text(
    image_paths,
    reader
)

print(ocr_docs)