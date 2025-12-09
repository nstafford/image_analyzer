# Image Analyzer

A Python-based image analysis application with a modern GUI for loading, viewing, and analyzing images. Features include metadata extraction and ML-based image segmentation.

## Features

- **Modern UI**: Built with tkinter and ttkbootstrap for a sleek, themed interface
- **Multi-format Support**: Load JPEG, PNG, BMP, TIFF, GIF, WebP images
- **Image Metrics**: Display resolution, channels, color space, bit depth, format, and file size
- **ML Segmentation**: Run pre-trained DeepLabV3 segmentation models with automatic device selection (CPU/GPU)
- **Smart System Detection**: Automatically detects available RAM, GPU, and VRAM to optimize model loading

## Technology Stack

- **UI Framework**: tkinter with ttkbootstrap
- **Image Processing**: PIL/Pillow
- **ML Framework**: PyTorch with torchvision
- **Computer Vision**: OpenCV
- **Numerical Computing**: NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image_analyzer.git
cd image_analyzer
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

1. **Load Image**: Click "Load Image" to select an image file
2. **View Metrics**: Image metadata displays automatically in the right panel
3. **Run ML Segmentation**: Click "Run ML Segmentation" to:
   - Check system capabilities (RAM, GPU, VRAM)
   - Load the segmentation model to the optimal device
   - Run inference and display colored mask overlay

## Project Structure

```
image_analyzer/
├── main.py                 # Application entry point
├── config.py               # Configuration settings
├── requirements.txt        # Project dependencies
├── gui/
│   ├── __init__.py
│   └── main_window.py     # Main application window
├── models/
│   ├── __init__.py
│   └── segmentation.py    # ML segmentation model wrapper
└── utils/
    ├── __init__.py
    ├── image_loader.py    # Image file operations
    ├── metrics.py         # Image metadata extraction
    └── system_checker.py  # System capability detection
```

## System Requirements

- Python 3.8+
- RAM: 4GB minimum, 8GB+ recommended for ML features
- GPU: Optional (CUDA-enabled GPU will be auto-detected and used)

## Future Enhancements

- Batch processing for multiple images
- Additional analysis metrics (histogram, color distribution)
- Multiple ML model options
- Export analysis results to CSV/JSON
- Image preprocessing tools (crop, rotate, adjust)
- Comparison view for multiple images

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
