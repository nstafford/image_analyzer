"""ML segmentation model wrapper with automatic device selection."""

from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from utils.system_checker import get_system_info
from models.model_registry import ModelRegistry, ModelConfig


class SegmentationModel:
    """Wrapper for segmentation model with automatic device selection."""
    
    def __init__(self, model_config: Optional[ModelConfig] = None):
        """Initialize the segmentation model.
        
        Args:
            model_config: Configuration for the model to load. If None, uses default DeepLabV3.
        """
        self.model = None
        self.device = None
        self.system_info = None
        
        # Use provided config or default to PASCAL VOC DeepLabV3
        if model_config is None:
            model_config = ModelRegistry.get_model_by_id("deeplabv3_pascal")
        
        self.model_config = model_config
        self.model_name = model_config.display_name
        self.weights_source = model_config.weights_source
        self.num_classes = model_config.num_classes
        self.class_names = model_config.class_names
        self.color_palette = model_config.color_palette
        self.num_parameters = 0
        self.model_size_mb = 0.0
        
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
            
            # Step 3: Load model using config's loader
            if self.model_config.model_loader is None:
                return False, "Model loader not configured"
            
            # For custom weights, pass the path to the loader
            if self.weights_source in ["CUSTOM", "BUNDLED"] and self.model_config.custom_weights_path:
                print(f"Loading model with weights from {self.model_config.custom_weights_path}...")
                self.model = self.model_config.model_loader(self.model_config.custom_weights_path)
            else:
                # Default weights or no weights
                self.model = self.model_config.model_loader()
            
            if self.model is None:
                return False, "Model loader returned None"
            
            self.model.eval()
            
            # Step 4: Move model to device
            self.model = self.model.to(self.device)
            
            # Step 5: Calculate model statistics
            self.num_parameters = sum(p.numel() for p in self.model.parameters())
            self.model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            
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
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model details
        """
        return {
            'name': self.model_name,
            'num_parameters': self.num_parameters,
            'size_mb': self.model_size_mb,
            'weights': self.weights_source,
            'device': str(self.device) if self.device else 'Not loaded'
        }
    
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
            
            # Handle different output formats based on number of classes
            if self.num_classes == 1:
                # Single channel output (binary segmentation with sigmoid)
                # Apply sigmoid and threshold at 0.5
                output_predictions = (torch.sigmoid(output[0]) > 0.5).cpu().numpy().astype(np.uint8)
            else:
                # Multi-class output (use argmax)
                output_predictions = output.argmax(0).cpu().numpy()
            
            return output_predictions
            
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return None
    
    def create_colored_mask(self, mask: np.ndarray) -> Image.Image:
        """
        Create a colored visualization of the segmentation mask.
        
        Args:
            mask: Segmentation mask array (0=background, 1=foreground for binary)
            
        Returns:
            PIL Image with colored mask
        """
        # Get color palette for current model
        palette = self._get_palette()
        
        # Create RGB image
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        # For single class models, mask is binary (0 or 1)
        # Apply color only to foreground (value=1)
        if self.num_classes == 1:
            # Background stays black (0,0,0), foreground gets the color
            colored_mask[mask == 1] = palette[0]
        else:
            # Multi-class: color each class
            for class_id in range(self.num_classes):
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
    
    def _get_palette(self):
        """Get color palette for the current model."""
        return self.color_palette
    
    def get_class_names(self):
        """Get class names for the current model."""
        return self.class_names
    
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
        palette = self._get_palette()
        
        stats = []
        
        # For single-class models, show background and foreground
        if self.num_classes == 1:
            # Background (0)
            bg_pixels = np.sum(mask == 0)
            bg_percentage = (bg_pixels / total_pixels) * 100
            stats.append({
                'class_id': 0,
                'name': 'background',
                'color': (0, 0, 0),
                'percentage': bg_percentage
            })
            
            # Foreground (1) - the actual class
            fg_pixels = np.sum(mask == 1)
            fg_percentage = (fg_pixels / total_pixels) * 100
            stats.append({
                'class_id': 1,
                'name': class_names[0],
                'color': tuple(palette[0]),
                'percentage': fg_percentage
            })
        else:
            # Multi-class: show all classes with their percentages
            for class_id in range(self.num_classes):
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
