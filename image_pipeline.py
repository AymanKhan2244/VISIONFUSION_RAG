import fitz
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from  langchain_community.vectorstores import FAISS
from chromadb.config import Settings



def extract_images(pdf_path:str):
    
    doc = fitz.open(pdf_path)
    os.makedirs("extracted_images", exist_ok=True)
    image_paths = []

    for page_index in range(len(doc)):

        page = doc[page_index]

    image_list = page.get_images(full=True)

    for img_index, img in enumerate(image_list):

        xref = img[0]

        base_image = doc.extract_image(xref)

        image_bytes = base_image["image"]

        image_ext = base_image["ext"]

        image_filename = f"extracted_images/page_{page_index+1}_img_{img_index+1}.{image_ext}"

        with open(image_filename, "wb") as f:
            f.write(image_bytes)

        image_paths.append(image_filename)

        print(f"Total Images Extracted: {len(image_paths)}")
    
    
    return image_paths




def load_blip_model():

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return processor, model


def generate_captions(
    image_paths,
    processor,
    model
):

    captions = []

    for image_path in image_paths:

        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = processor(
            image,
            return_tensors="pt"
        )

        output = model.generate(
            **inputs
        )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        captions.append(caption)

    return captions