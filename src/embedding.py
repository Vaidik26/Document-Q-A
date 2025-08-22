from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor


class SiglipEmbedding:
    """Wrapper around a SigLIP model to provide text and image embeddings.

    Loads the model and processor once, and exposes simple methods for
    generating L2-normalized embeddings.
    """

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384") -> None:
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def embed_image(self, image_data) -> "torch.Tensor":
        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            image = image_data

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

    def embed_text(self, text_data: str) -> "torch.Tensor":
        inputs = self.processor(
            text=text_data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()


# Create a shared embedder instance for simple usage across modules
EMBEDDER = SiglipEmbedding()


