import os
import streamlit as st

from ingestion import ingest_pdf
from qdrant_loader import load_vectorstore

from reranker import rerank_documents
from llm import generate_answer


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="VisionFusion RAG",
    page_icon="🤖",
    layout="wide"
)

# =====================================
# TITLE
# =====================================

st.title("🤖 VisionFusion RAG")

st.markdown("""
Multimodal RAG System using:

- Qdrant Cloud
- Hybrid Search
- OCR
- BLIP Captioning
- Groq LLM
- Reranking
""")

# =====================================
# SESSION STATE
# =====================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================
# PDF UPLOAD
# =====================================

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    os.makedirs(
        "data/pdfs",
        exist_ok=True
    )

    pdf_path = os.path.join(
        "data/pdfs",
        uploaded_file.name
    )

    with open(
        pdf_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    if st.sidebar.button(
        "Process PDF"
    ):

        with st.spinner(
            "Processing PDF..."
        ):

            ingest_pdf(
                pdf_path
            )

        st.success(
            "PDF Processed Successfully!"
        )

# =====================================
# LOAD VECTOR DB
# =====================================

if st.sidebar.button(
    "Load Knowledge Base"
):

    st.session_state.vectorstore = (
        load_vectorstore()
    )

    st.success(
        "Knowledge Base Loaded!"
    )

# =====================================
# CHAT HISTORY
# =====================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# =====================================
# CHAT INPUT
# =====================================

query = st.chat_input(
    "Ask a question..."
)

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(query)

    if (
        st.session_state.vectorstore
        is None
    ):

        st.error(
            "Load Knowledge Base First!"
        )

    else:

        with st.spinner(
            "Thinking..."
        ):

            retriever = (
                st.session_state
                .vectorstore
                .as_retriever(
                    search_kwargs={
                        "k": 15
                    }
                )
            )

            results = (
                retriever.invoke(
                    query
                )
            )

            reranked_results = results[:5]

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc
                    in reranked_results
                ]
            )

            answer = (
                generate_answer(
                    query,
                    context
                )
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message(
            "assistant"
        ):

            st.markdown(answer)

        # =====================
        # SOURCES
        # =====================

        st.subheader(
            "Retrieved Sources"
        )

        for idx, doc in enumerate(
            reranked_results,
            start=1
        ):

            with st.expander(
                f"Source {idx}"
            ):

                st.write(
                    doc.page_content
                )

                st.json(
                    doc.metadata
                )

                if (
                    "image_path"
                    in doc.metadata
                ):

                    try:

                        st.image(
                            doc.metadata[
                                "image_path"
                            ]
                        )

                    except:

                        pass