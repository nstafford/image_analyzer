"""Registry of available segmentation models and their configurations."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import torch


@dataclass
class ModelConfig:
    """Configuration for a segmentation model."""
    
    id: str
    name: str
    display_name: str
    architecture: str
    num_classes: int
    class_names: List[str]
    color_palette: List[Tuple[int, int, int]]
    weights_source: str  # "DEFAULT", "CUSTOM", or path
    custom_weights_path: Optional[str] = None
    model_loader: Optional[Callable] = None  # Function to load the model
    description: str = ""
    input_size_recommendation: Optional[Tuple[int, int]] = None
    
    def validate_weights_file(self, weights_path: str) -> Tuple[bool, str]:
        """
        Validate that custom weights file exists and matches model architecture.
        
        Args:
            weights_path: Path to weights file
            
        Returns:
            Tuple of (valid: bool, message: str)
        """
        import os
        
        if not os.path.exists(weights_path):
            return False, f"Weights file not found: {weights_path}"
        
        try:
            # Load weights to check structure
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Basic validation - check if it's a state dict
            if not isinstance(state_dict, dict):
                return False, "Invalid weights file format"
            
            # Could add more specific checks here based on architecture
            # For now, just verify it's loadable
            
            return True, "Weights file validated successfully"
            
        except Exception as e:
            return False, f"Error loading weights: {str(e)}"


class ModelRegistry:
    """Registry of available segmentation models."""
    
    # PASCAL VOC palette (21 classes)
    PASCAL_PALETTE = [
        (0, 0, 0),       # background
        (128, 0, 0),     # aeroplane
        (0, 128, 0),     # bicycle
        (128, 128, 0),   # bird
        (0, 0, 128),     # boat
        (128, 0, 128),   # bottle
        (0, 128, 128),   # bus
        (128, 128, 128), # car
        (64, 0, 0),      # cat
        (192, 0, 0),     # chair
        (64, 128, 0),    # cow
        (192, 128, 0),   # dining table
        (64, 0, 128),    # dog
        (192, 0, 128),   # horse
        (64, 128, 128),  # motorbike
        (192, 128, 128), # person
        (0, 64, 0),      # potted plant
        (128, 64, 0),    # sheep
        (0, 192, 0),     # sofa
        (128, 192, 0),   # train
        (0, 64, 128)     # tv/monitor
    ]
    
    PASCAL_CLASS_NAMES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow",
        "dining table", "dog", "horse", "motorbike", "person",
        "potted plant", "sheep", "sofa", "train", "tv/monitor"
    ]
    
    # Binary segmentation palette (2 classes)
    BINARY_PALETTE = [
        (0, 0, 0),       # background
        (255, 0, 0)      # foreground/object
    ]
    
    BINARY_CLASS_NAMES = [
        "background",
        "object"
    ]
    
    @staticmethod
    def _load_2class_model_with_weights(weights_path: str):
        """
        Load custom lung segmentation model with weights.
        Model has 1 output class (lung foreground only).
        
        Args:
            weights_path: Path to the weights file
            
        Returns:
            Loaded model
        """
        from torchvision.models.segmentation import deeplabv3_resnet50
        from torchvision.models.segmentation.deeplabv3 import DeepLabHead
        import torch.nn as nn
        import os
        
        # Load the weights/checkpoint first to avoid downloading backbone
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Handle different save formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Saved with metadata
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict):
            # Saved as raw state_dict
            state_dict = checkpoint
        else:
            # Saved as complete model
            if hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
            else:
                raise ValueError("Cannot extract state_dict from checkpoint")
        
        # Create base model (will download ResNet50 if needed)
        # Use weights='DEFAULT' to get the backbone, we'll override the classifier
        model = deeplabv3_resnet50(weights='DEFAULT')
        
        # Modify classifier to have 1 output class (matching your training setup)
        model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        
        # Load the trained weights (this will override classifier and any trained backbone weights)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        return model
    
    @staticmethod
    def get_available_models() -> List[ModelConfig]:
        """
        Get list of all available models.
        
        Returns:
            List of ModelConfig objects
        """
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        import os
        
        # Get paths to bundled weights
        models_dir = os.path.dirname(os.path.abspath(__file__))
        bundled_weights = os.path.join(models_dir, "weights", "best_deeplab_lung_seg.pth")
        
        models = [
            # DeepLabV3 with default PASCAL VOC weights
            ModelConfig(
                id="deeplabv3_pascal",
                name="DeepLabV3-MobileNetV3",
                display_name="DeepLabV3 (MobileNetV3-Large) - PASCAL VOC",
                architecture="DeepLabV3",
                num_classes=21,
                class_names=ModelRegistry.PASCAL_CLASS_NAMES,
                color_palette=ModelRegistry.PASCAL_PALETTE,
                weights_source="DEFAULT",
                model_loader=lambda: deeplabv3_mobilenet_v3_large(weights='DEFAULT'),
                description="Pre-trained on PASCAL VOC dataset with 21 classes. "
                           "Good for general object segmentation.",
                input_size_recommendation=(512, 512)
            ),
            
            # DeepLabV3 with bundled lung segmentation weights
            ModelConfig(
                id="deeplabv3_lung_seg",
                name="DeepLabV3-LungSeg",
                display_name="DeepLabV3 (ResNet50) - Lung Segmentation",
                architecture="DeepLabV3-ResNet50",
                num_classes=1,
                class_names=["lung"],
                color_palette=[(255, 0, 0)],  # Red for lung
                weights_source="BUNDLED",
                custom_weights_path=bundled_weights if os.path.exists(bundled_weights) else None,
                model_loader=lambda wp=bundled_weights: ModelRegistry._load_2class_model_with_weights(wp) if wp and os.path.exists(wp) else None,
                description="Custom lung segmentation model (ResNet50 backbone, 1 output class) with pre-loaded weights.",
                input_size_recommendation=(512, 512)
            ),
            
            # DeepLabV3 with custom user-provided weights
            ModelConfig(
                id="deeplabv3_custom_binary",
                name="DeepLabV3-Custom-Binary",
                display_name="DeepLabV3 (ResNet50) - Custom Binary",
                architecture="DeepLabV3-ResNet50",
                num_classes=2,
                class_names=ModelRegistry.BINARY_CLASS_NAMES,
                color_palette=ModelRegistry.BINARY_PALETTE,
                weights_source="CUSTOM",
                custom_weights_path=None,  # User will specify
                model_loader=lambda wp: ModelRegistry._load_2class_model_with_weights(wp) if wp else None,
                description="Custom 2-class segmentation model (ResNet50 backbone). "
                           "Requires user to load custom weights file.",
                input_size_recommendation=(512, 512)
            ),
        ]
        
        return models
    
    @staticmethod
    def get_model_by_id(model_id: str) -> Optional[ModelConfig]:
        """
        Get model configuration by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelConfig object or None if not found
        """
        models = ModelRegistry.get_available_models()
        for model in models:
            if model.id == model_id:
                return model
        return None
    
    @staticmethod
    def get_model_display_names() -> List[Tuple[str, str]]:
        """
        Get list of (model_id, display_name) tuples for UI display.
        
        Returns:
            List of tuples (id, display_name)
        """
        models = ModelRegistry.get_available_models()
        return [(m.id, m.display_name) for m in models]
