import os
from pathlib import Path
from typing import List
from pdf2image import convert_from_path
import tempfile
from PIL import Image


def load_file_as_images(file_path: str) -> List[str]:
    """
    Loads a file (PDF or Image) and converts it to a list of temporary image file paths.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Create a unique temp directory for this run?
    # Or just use mkdtemp. The caller is responsible for cleanup or we just let OS handle it (eventually).
    # Ideally we return a context manager or just the paths and a cleanup function.
    # For simplicity of the "pipeline" style, let's just create them and return paths.

    temp_dir = tempfile.mkdtemp(prefix="oba_")
    image_paths = []

    if path.suffix.lower() == ".pdf":
        try:
            images = convert_from_path(str(path))
            for i, img in enumerate(images):
                img_path = os.path.join(temp_dir, f"page_{i}.jpg")
                img.save(img_path, "JPEG")
                image_paths.append(img_path)
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF: {e}")
    elif path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
        try:
            img = Image.open(path)
            img_path = os.path.join(temp_dir, "page_0.jpg")
            img.convert("RGB").save(img_path, "JPEG")
            image_paths.append(img_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return image_paths


def cleanup_images(image_paths: List[str]):
    """Removes temporary image files."""
    for path in image_paths:
        if "oba_" in path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    # Basic cleanup, potentially leaving empty temp dirs but acceptable for MVP CLI
