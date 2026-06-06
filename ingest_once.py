from ingestion import ingest_pdf

PDF_PATH = (
    "data/pdfs/sample.pdf"
)

ingest_pdf(
    PDF_PATH
)

print(
    "Knowledge Base Built!"
)