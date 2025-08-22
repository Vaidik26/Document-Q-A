from src.prompt import create_multimodal_message
from src.utils import get_llm
from src.vector_embedding_store import build_vector_store
from src.embedding import EMBEDDER
import os
from dotenv import load_dotenv


load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


llm = get_llm()


def prepare_session(pdf_path: str):
    """Build per-PDF vector store and return a retrieval callable and state."""
    all_docs, image_data_store, vector_store = build_vector_store(pdf_path)

    def retrieve(query: str, k: int = 5):
        query_embedding = EMBEDDER.embed_text(query)
        return vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)

    return retrieve, image_data_store


def multimodal_pdf_rag_pipeline(query, retrieve, image_data_store):
    """Main pipeline for multimodal RAG using provided session retrieve and image data."""
    # Retrieve relevant documents
    context_docs = retrieve(query, k=5)

    # Create multimodal message
    message = create_multimodal_message(query, context_docs, image_data_store)

    # Get response from LLM
    response = llm.invoke([message])

    # Print retrieved context info
    print(f"\nRetrieved {len(context_docs)} documents:")
    for doc in context_docs:
        doc_type = doc.metadata.get("type", "unknown")
        page = doc.metadata.get("page", "?")
        if doc_type == "text":
            preview = (
                doc.page_content[:100] + "..."
                if len(doc.page_content) > 100
                else doc.page_content
            )
            print(f"  - Text from page {page}: {preview}")
        else:
            print(f"  - Image from page {page}")
    print("\n")

    return response.content
