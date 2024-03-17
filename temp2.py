from transformers import CLIPProcessor, CLIPModel
import torch 
import numpy as np


def get_text_embedding(text):
    # Check for empty or None text
    if not text:
        raise ValueError("Text cannot be empty or None.")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16-224")

    # Load CLIP processor
    tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16-224")

    # Preprocess the text
    inputs = tokenizer(text=text, return_tensors="pt")

    # Get the text embedding
    with torch.no_grad():
        outputs = model(**inputs)
    text_embeds = outputs.text_embeds

    # Convert the embedding to NumPy array and reshape to desired size
    text_embedding = text_embeds.cpu().detach().numpy().reshape(1, -1)

    return text_embedding

# Example usage
text = "This is an example text."
embedding = get_text_embedding(text)
print(f"Text embedding shape: {embedding.shape}")
print(f"Text embedding: {embedding[0]}")
