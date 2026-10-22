"""Download the official SHA-256-verified OpenAI CLIP ViT-L/14 initialization."""
from pathlib import Path
import clip

root = Path(__file__).resolve().parents[1] / "weights"
print(clip.clip._download(clip.clip._MODELS["ViT-L/14"], str(root)))
