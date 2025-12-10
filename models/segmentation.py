"""ML segmentation model wrapper with automatic device selection."""

from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from utils.system_checker import get_system_info


class SegmentationModel:
    """Wrapper for segmentation model with automatic device selection."""
    
    def __init__(self):
        """Initialize the segmentation model."""
        self.model = None
        self.device = None
        self.system_info = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self) -> Tuple[bool, str]:
        """
        Check system capabilities and load the model to appropriate device.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Step 1: Check system capabilities
            print("Step 1: Checking system capabilities...")
            self.system_info = get_system_info()
            
            device_name = self.system_info['recommended_device']
            print(f"Recommended device: {device_name}")
            print(f"RAM available: {self.system_info['ram_available_mb']} MB")
            
            if self.system_info['gpu_available']:
                print(f"GPU detected: {self.system_info['gpu_name']}")
                if self.system_info['gpu_vram_mb'] > 0:
                    print(f"VRAM: {self.system_info['gpu_vram_mb']:.0f} MB")
            else:
                print("No GPU detected, using CPU")
            
            # Step 2: Set device
            if device_name == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif device_name == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
            
            print(f"Step 2: Loading model to {self.device}...")
            
            # Step 3: Load pre-trained model
            self.model = deeplabv3_mobilenet_v3_large(weights='DEFAULT')
            self.model.eval()
            
            # Step 4: Move model to device
            self.model = self.model.to(self.device)
            
            success_msg = f"Model loaded successfully to {self.device}\n"
            
            # Determine what hardware is being used
            if self.device.type == 'cuda':
                success_msg += f"Using: {self.system_info['gpu_name']} (CUDA)"
            elif self.device.type == 'mps':
                success_msg += f"Using: {self.system_info['gpu_name']} (Metal Performance Shaders)"
            else:
                # CPU mode
                if self.system_info['gpu_available']:
                    # GPU detected but not used
                    success_msg += f"Using: CPU (GPU detected: {self.system_info['gpu_name']}, but no CUDA support)"
                else:
                    success_msg += "Using: CPU (No GPU detected)"
            
            print("Model loading complete!")
            return True, success_msg
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def segment_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Run segmentation on an image.
        
        Args:
            image: PIL Image to segment
            
        Returns:
            Segmentation mask as numpy array, or None if failed
        """
        if self.model is None or self.device is None:
            print("Model not loaded. Call load_model() first.")
            return None
        
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess
            input_tensor = self.transform(image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_batch)['out'][0]
            
            # Get predicted class for each pixel
            output_predictions = output.argmax(0).cpu().numpy()
            
            return output_predictions
            
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return None
    
    def create_colored_mask(self, mask: np.ndarray) -> Image.Image:
        """
        Create a colored visualization of the segmentation mask.
        
        Args:
            mask: Segmentation mask array
            
        Returns:
            PIL Image with colored mask
        """
        # Create color palette (21 classes for COCO)
        palette = self._get_pascal_palette()
        
        # Create RGB image
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for class_id in range(21):
            colored_mask[mask == class_id] = palette[class_id]
        
        return Image.fromarray(colored_mask)
    
    def overlay_mask(self, image: Image.Image, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
        """
        Overlay segmentation mask on original image.
        
        Args:
            image: Original PIL Image
            mask: Segmentation mask array
            alpha: Transparency of mask overlay (0-1)
            
        Returns:
            PIL Image with mask overlay
        """
        colored_mask = self.create_colored_mask(mask)
        
        # Resize mask to match image if needed
        if colored_mask.size != image.size:
            colored_mask = colored_mask.resize(image.size, Image.Resampling.NEAREST)
        
        # Convert image to RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        colored_mask = colored_mask.convert('RGBA')
        
        # Blend images
        blended = Image.blend(image, colored_mask, alpha)
        
        return blended.convert('RGB')
    
    @staticmethod
    def _get_pascal_palette():
        """Get color palette for PASCAL VOC classes."""
        palette = [
            [0, 0, 0],       # background
            [128, 0, 0],     # aeroplane
            [0, 128, 0],     # bicycle
            [128, 128, 0],   # bird
            [0, 0, 128],     # boat
            [128, 0, 128],   # bottle
            [0, 128, 128],   # bus
            [128, 128, 128], # car
            [64, 0, 0],      # cat
            [192, 0, 0],     # chair
            [64, 128, 0],    # cow
            [192, 128, 0],   # dining table
            [64, 0, 128],    # dog
            [192, 0, 128],   # horse
            [64, 128, 128],  # motorbike
            [192, 128, 128], # person
            [0, 64, 0],      # potted plant
            [128, 64, 0],    # sheep
            [0, 192, 0],     # sofa
            [128, 192, 0],   # train
            [0, 64, 128]     # tv/monitor
        ]
        return palette
    
    @staticmethod
    def get_class_names():
        """Get class names for PASCAL VOC dataset."""
        return [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "dining table",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "tv/monitor"
        ]
    
    def get_segmentation_stats(self, mask: np.ndarray) -> list:
        """
        Calculate statistics for each class in the segmentation mask.
        
        Args:
            mask: Segmentation mask array
            
        Returns:
            List of dictionaries with class_id, name, color, and percentage
        """
        total_pixels = mask.size
        class_names = self.get_class_names()
        palette = self._get_pascal_palette()
        
        stats = []
        
        # Show all 21 classes with their percentages
        for class_id in range(len(class_names)):
            class_pixels = np.sum(mask == class_id)
            percentage = (class_pixels / total_pixels) * 100
            
            stats.append({
                'class_id': int(class_id),
                'name': class_names[class_id],
                'color': tuple(palette[class_id]) if class_id < len(palette) else (128, 128, 128),
                'percentage': percentage
            })
        
        # Sort by percentage descending
        stats.sort(key=lambda x: x['percentage'], reverse=True)
        
        return stats
