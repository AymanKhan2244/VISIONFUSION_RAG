from langchain_text_splitters import RecursiveCharacterTextSplitter




document = "Attention.pdf"
def text_splitter(documnent):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_overlap=50,
    chunk_size=500
)
    chunks = text_splitter.split_documents(document)

    return len(chunks)

