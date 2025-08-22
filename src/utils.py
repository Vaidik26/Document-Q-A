import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor


# ---------- Model & Processor Loader ----------

def load_model_and_processor(model_name: str):
    """Load model and processor for given model_name"""
    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

# ---------- File Loader ----------

def load_pdf(file_path):
    doc = fitz.open(file_path)
    return doc


# ---------- Embedding Functions ----------

# ---------- Embedding Image ----------

def embed_image(image_data, model, processor):
    """Embed image using SigLip"""
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        # Normalize embedding to unit vector
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

# ---------- Embedding Text ----------

def embed_text(text_data, model, processor):
    """Embed image using SigLip"""
    inputs = processor(
        text=text_data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64 # Max token length
    )

    with torch.no_grad():
        features = model.get_text_features(**inputs)
        # Normalize embedding to unit vector
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()