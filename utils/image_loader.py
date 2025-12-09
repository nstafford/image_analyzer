"""Image loading and validation utilities."""

from typing import Optional
from PIL import Image
import os


def load_image(filepath: str) -> Optional[Image.Image]:
    """
    Load an image from the specified filepath.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        PIL Image object if successful, None otherwise
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        image = Image.open(filepath)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def validate_image_format(filepath: str) -> bool:
    """
    Validate if the file is a supported image format.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        True if the format is supported, False otherwise
    """
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']
    _, ext = os.path.splitext(filepath)
    return ext.lower() in supported_extensions
