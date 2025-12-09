"""Image metadata extraction utilities."""

from typing import Dict, Any
from PIL import Image
import os


def extract_image_metrics(image: Image.Image, filepath: str) -> Dict[str, Any]:
    """
    Extract metadata and metrics from an image.
    
    Args:
        image: PIL Image object
        filepath: Path to the image file
        
    Returns:
        Dictionary containing image metrics
    """
    metrics = {}
    
    # Resolution
    metrics['width'] = image.width
    metrics['height'] = image.height
    metrics['resolution'] = f"{image.width} x {image.height}"
    
    # Format
    metrics['format'] = image.format if image.format else "Unknown"
    
    # Mode and channels
    metrics['mode'] = image.mode
    if image.mode == 'L':
        metrics['channels'] = 1
        metrics['color_space'] = "Grayscale"
    elif image.mode == 'RGB':
        metrics['channels'] = 3
        metrics['color_space'] = "RGB"
    elif image.mode == 'RGBA':
        metrics['channels'] = 4
        metrics['color_space'] = "RGBA"
    elif image.mode == 'CMYK':
        metrics['channels'] = 4
        metrics['color_space'] = "CMYK"
    else:
        metrics['channels'] = len(image.getbands())
        metrics['color_space'] = image.mode
    
    # Bit depth
    if hasattr(image, 'bits'):
        metrics['bit_depth'] = image.bits
    else:
        # Estimate bit depth from mode
        if image.mode in ['1']:
            metrics['bit_depth'] = 1
        elif image.mode in ['L', 'P']:
            metrics['bit_depth'] = 8
        elif image.mode in ['RGB', 'RGBA', 'CMYK']:
            metrics['bit_depth'] = 8
        elif image.mode in ['I', 'F']:
            metrics['bit_depth'] = 32
        else:
            metrics['bit_depth'] = "Unknown"
    
    # File size
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        metrics['file_size_bytes'] = file_size
        metrics['file_size'] = format_file_size(file_size)
    else:
        metrics['file_size'] = "Unknown"
    
    return metrics


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"
