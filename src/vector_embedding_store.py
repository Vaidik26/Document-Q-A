from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
import base64
import io
import numpy as np
from langchain_community.vectorstores import FAISS
from src.utils import load_pdf, embed_text, embed_image

doc = load_pdf("data")

def storing_vector_embed():
    
    all_docs = []
    all_embeddings = []
    image_data_store = {}
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    for i,page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            temp_doc = Document(page_content=text, metadata={"page":i, "type":"text"})
            text_chunk = splitter.split_documents([temp_doc])

            for chunk in text_chunk:
                embeddings = embed_text(chunk.page_content)
                all_embeddings.append(embeddings)
                all_docs.append(chunk)

        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Create unique identifier
                image_id = f"page_{i}_img_{img_index}"

                # Store image as base64 for later use with Model
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = img_base64

                # Embed image using Model
                embedding = embed_image(pil_image)
                all_embeddings.append(embedding)

                # Create document for image
                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": i, "type": "image", "image_id": image_id}
                )
                all_docs.append(image_doc)

            except Exception as e:
                print(f"Error processing image {img_index} on page {i}: {e}")
                continue

    doc.close()
    
    # Create unified FAISS vector store with CLIP embeddings
    embeddings_array = np.array(all_embeddings)
    
    # Create custom FAISS index since we have precomputed embeddings
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
        embedding=None,  # We're using precomputed embeddings
        metadatas=[doc.metadata for doc in all_docs]
    )
    return all_docs, all_embeddings, image_data_store, vector_store