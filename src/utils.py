import fitz  # PyMuPDF
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI


def get_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return llm


# ---------- Model & Processor Loader ----------


def load_model_and_processor(model_name: str):
    """Deprecated: model loading is handled by src.embedding.EMBEDDER"""
    raise NotImplementedError(
        "Use src.embedding.EMBEDDER instead of loading ad-hoc models"
    )


# ---------- File Loader ----------


def load_pdf(file_path):
    doc = fitz.open(file_path)
    return doc


# ---------- Query Search ----------


def retrieve_multimodal(*_args, **_kwargs):
    """Deprecated: use session-based retrieval in src.rag_pipeline"""
    raise NotImplementedError("Use functions in src.rag_pipeline for retrieval")


__all__ = [
    "get_llm",
    "load_pdf",
]
