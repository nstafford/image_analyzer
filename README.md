# ml-seg-viz

A Python-based image analysis application with ML segmentation capabilities. Load images, select ML models, and visualize segmentation results with detailed metrics and interactive overlay controls.

## Features

- **Modern UI**: Built with tkinter and ttkbootstrap for a sleek, themed dark interface
- **Multi-format Support**: Load JPEG, PNG, BMP, TIFF, GIF, WebP images
- **Image Metrics**: Display resolution, channels, color space, bit depth, format, and file size
- **System Information**: Real-time display of RAM, GPU, and VRAM availability
- **Model Management**: 
  - Load ML models independently from image analysis
  - Model selection dialog (currently DeepLabV3 MobileNetV3)
  - View model details: parameters, size, weights source, device
- **ML Segmentation**: 
  - Apply loaded models to images for semantic segmentation
  - Automatic device selection (CPU/CUDA/MPS)
  - 21-class PASCAL VOC segmentation
- **Interactive Visualization**:
  - Colored mask overlay with adjustable transparency
  - Toggle overlay on/off with checkbox
  - Segmentation results panel showing all classes with color swatches and percentages
  - Responsive window resizing with image scaling
- **Efficient Workflow**: Load model once, analyze multiple images without reloading

## Technology Stack

- **UI Framework**: tkinter with ttkbootstrap
- **Image Processing**: PIL/Pillow
- **ML Framework**: PyTorch with torchvision
- **Computer Vision**: OpenCV
- **Numerical Computing**: NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nstafford/ml-seg-viz.git
cd ml-seg-viz
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/macOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python main.py
```

### Basic Workflow

1. **Load ML Model**: Click "Load ML Model" to:
   - Select a segmentation model from the dialog
   - Model loads once and displays info (name, parameters, size, weights, device)
   - Status updates in toolbar showing loaded model name
2. **Load Image**: Click "Load Image" to select an image file
   - Image metrics display automatically in the right panel
   - Previous segmentation results clear for new image
3. **Apply Model**: Click "Apply Model to Image" to:
   - Run segmentation on the current image
   - Display colored mask overlay
   - Show all 21 PASCAL VOC classes with percentages
4. **Toggle Overlay**: Use the checkbox to switch between original image and segmented view
5. **Load More Images**: Load additional images and apply the same model without reloading

## Project Structure

```
ml-seg-viz/
├── main.py                 # Application entry point
├── config.py               # Configuration settings (theme, window size, formats)
├── requirements.txt        # Project dependencies
├── gui/
│   ├── __init__.py
│   └── main_window.py     # Main application window with all UI components
├── models/
│   ├── __init__.py
│   └── segmentation.py    # ML segmentation model wrapper with device detection
└── utils/
    ├── __init__.py
    ├── image_loader.py    # Image file loading and validation
    ├── metrics.py         # Image metadata extraction
    └── system_checker.py  # System capability detection (RAM, GPU, VRAM)
```

## UI Layout

- **Top Toolbar**: Load Image, Load ML Model, model status label, Apply Model to Image
- **Left Panel**: Image display area (resizable, maintains aspect ratio)
- **Right Panel** (fixed 320px width):
  - Image Metrics: Resolution, channels, color space, bit depth, format, file size
  - System Information: Platform, RAM, GPU, VRAM (loads at startup in background)
  - Model Information: Name, parameters, size, weights, device
  - Segmentation Results: Scrollable list with color swatches, class names, percentages
  - Overlay Toggle: Checkbox to show/hide segmentation mask
- **Status Bar**: Real-time operation feedback

## System Requirements

- Python 3.8+
- RAM: 4GB minimum, 8GB+ recommended for ML features
- GPU: Optional (CUDA-enabled GPU will be auto-detected and used if available)
- OS: Windows, macOS, or Linux

## Segmentation Model

Currently uses **DeepLabV3 with MobileNetV3-Large backbone**:
- Pre-trained on PASCAL VOC dataset (21 classes)
- Classes: background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train, tv/monitor
- Lightweight and efficient for CPU inference
- Automatically downloads weights on first use (~17MB)

## Future Enhancements

- [ ] Batch processing for multiple images
- [ ] Additional segmentation models (U-Net, Mask R-CNN, SAM)
- [ ] Custom model loading from file
- [ ] Export segmentation masks as PNG
- [ ] Export analysis results to CSV/JSON
- [ ] Additional metrics (histogram, color distribution)
- [ ] Image preprocessing tools (crop, rotate, brightness/contrast)
- [ ] Side-by-side comparison view
- [ ] Overlay transparency slider
- [ ] Keyboard shortcuts

## License

GNU General Public License v3.0 (GPL-3.0)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
