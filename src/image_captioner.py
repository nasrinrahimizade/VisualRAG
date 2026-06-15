
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "E:\\huggingface_cache")

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_captioner():
    """
    Download (first run) and load the BLIP-2 processor and model into memory.
    The processor converts raw images into tensors the model understands.
    The model takes those tensors and generates a text description.
    Returns both as a tuple (processor, model) since you need both for captioning.
    Only call this once — loading takes ~30 seconds and you don't want to repeat it.
    """
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)

    return (processor, model)

def caption_image(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images= image, return_tensors='pt')
    inputs = inputs.to(device)
    generate_ids = model.generate(**inputs, max_new_tokens=50)
    generate_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
    return generate_text