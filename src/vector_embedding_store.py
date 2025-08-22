from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
import base64
import io
import numpy as np
from langchain_community.vectorstores import FAISS
import fitz
from src.embedding import EMBEDDER


def _load_pdf(file_path: str):
    return fitz.open(file_path)


def build_vector_store(pdf_path: str):
    """Load a PDF, extract text and images, embed them, and build a FAISS store.

    Returns:
        all_docs: list[Document]
        image_data_store: dict[str, str] mapping image_id -> base64 PNG data
        vector_store: FAISS vector store with precomputed embeddings
    """
    doc = _load_pdf(pdf_path)

    all_docs = []
    all_embeddings = []
    image_data_store = {}

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for i, page in enumerate(doc):
        text = page.get_text()
        if text and text.strip():
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
            text_chunk = splitter.split_documents([temp_doc])

            for chunk in text_chunk:
                embeddings = EMBEDDER.embed_text(chunk.page_content)
                all_embeddings.append(embeddings)
                all_docs.append(chunk)

        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                image_id = f"page_{i}_img_{img_index}"

                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = img_base64

                embedding = EMBEDDER.embed_image(pil_image)
                all_embeddings.append(embedding)

                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": i, "type": "image", "image_id": image_id},
                )
                all_docs.append(image_doc)

            except Exception as e:
                print(f"Error processing image {img_index} on page {i}: {e}")
                continue

    doc.close()

    embeddings_array = np.array(all_embeddings)

    vector_store = FAISS.from_embeddings(
        text_embeddings=[
            (d.page_content, emb) for d, emb in zip(all_docs, embeddings_array)
        ],
        embedding=None,
        metadatas=[d.metadata for d in all_docs],
    )

    return all_docs, image_data_store, vector_store
