---
applyTo: '**'
---

# Image Analyzer Application - Copilot Instructions

## Project Overview
This is a Python-based image analysis application with a modern GUI built using tkinter with ttkbootstrap theming. The application enables users to load images, view metadata, and perform ML-based image segmentation.

## Technology Stack
- **UI Framework**: tkinter with ttkbootstrap for modern, themed components
- **Image Processing**: PIL/Pillow for image loading and manipulation
- **ML Framework**: PyTorch or TensorFlow for segmentation models
- **Computer Vision**: OpenCV (cv2) for advanced image operations
- **Numerical Computing**: NumPy for array operations

## Code Style & Architecture

### Python Standards
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Use descriptive variable and function names (snake_case)
- Add docstrings to all classes and non-trivial functions
- Keep functions focused and modular (single responsibility principle)

### Project Structure
```
image_analyzer/
├── main.py                 # Application entry point
├── gui/
│   ├── __init__.py
│   ├── main_window.py     # Main application window
│   ├── image_viewer.py    # Image display widget
│   └── controls.py        # UI controls and buttons
├── models/
│   ├── __init__.py
│   ├── segmentation.py    # ML segmentation model wrapper
│   └── model_loader.py    # Model loading utilities
├── utils/
│   ├── __init__.py
│   ├── image_loader.py    # Image file operations
│   └── metrics.py         # Image metadata extraction
├── config.py              # Application configuration
└── requirements.txt       # Project dependencies
```

## UI/UX Guidelines

### ttkbootstrap Usage
- Use ttkbootstrap themed widgets (ttk.Button, ttk.Frame, ttk.Label, etc.)
- Apply consistent theme throughout (suggest: 'darkly', 'flatly', or 'cosmo')
- Initialize app with: `app = ttk.Window(themename="darkly")`
- Use ttkbootstrap's Bootstyle constants for styling (PRIMARY, SUCCESS, INFO, etc.)

### Layout Principles
- Use grid or pack geometry managers consistently within each frame
- Create separate frames for logical UI sections:
  - Top toolbar: file operations and controls
  - Left panel: image display area
  - Right panel: metrics and analysis results
  - Bottom panel: status bar
- Make image display area resizable and scrollable
- Ensure responsive layout that adapts to window resizing

### User Interactions
- Provide clear visual feedback for all operations (loading, processing)
- Show progress indicators for long-running operations (model inference)
- Display error messages in user-friendly dialog boxes
- Implement keyboard shortcuts for common operations (Ctrl+O for open)
- Add tooltips to explain functionality

## Image Handling

### Supported Formats
Support common image formats: JPEG, PNG, BMP, TIFF, GIF, WebP
- Use PIL.Image.open() for loading
- Handle format-specific quirks (RGBA vs RGB, transparency)
- Validate file extensions and mime types

### Image Display
- Scale images to fit display area while maintaining aspect ratio
- Implement zoom functionality (zoom in/out, fit to window)
- Use PIL.ImageTk.PhotoImage for tkinter compatibility
- Update display efficiently when switching between original and segmented views

### Metadata Extraction
Display the following metrics:
- **Resolution**: Width x Height in pixels
- **Channels**: Number of color channels (1=Grayscale, 3=RGB, 4=RGBA)
- **Bit Depth**: Bits per pixel/channel
- **Format**: File format (JPEG, PNG, etc.)
- **File Size**: Display in human-readable format (KB, MB)
- **Color Space**: RGB, CMYK, Grayscale, etc.

## ML Segmentation Features

### Model Integration
- Use pre-trained segmentation models (e.g., DeepLab, U-Net, Mask R-CNN)
- Implement async/threaded execution to prevent UI freezing
- Cache model in memory after first load to improve performance
- Provide model selection if multiple models are available

### Segmentation Display
- Overlay segmentation masks on original image with adjustable transparency
- Use distinct colors for different segmentation classes
- Allow toggling between original, mask-only, and overlay views
- Display class labels and confidence scores if available
- Provide mask export functionality (save segmented image)

### Performance Considerations
- Resize large images before inference if model has size limitations
- Display processing time for transparency
- Consider GPU acceleration (CUDA) if available
- Implement batch processing for multiple images (future feature)

## Error Handling
- Wrap file I/O operations in try-except blocks
- Validate image files before processing
- Handle model loading failures gracefully
- Provide informative error messages to users
- Log errors to console for debugging

## Dependencies Management
Key packages in requirements.txt:
```
ttkbootstrap>=1.10.1
Pillow>=10.0.0
numpy>=1.24.0
opencv-python>=4.8.0
torch>=2.5.0  # or tensorflow>=2.13.0
torchvision>=0.20.0  # for pre-trained models
```

## Testing Considerations
- Test with various image sizes (small to very large)
- Test with different image formats and color spaces
- Test edge cases: corrupted files, unsupported formats
- Verify memory usage with large images
- Test UI responsiveness during model inference

## Future Enhancements
Consider these when writing extensible code:
- Batch processing multiple images
- Additional analysis metrics (histogram, color distribution)
- Multiple ML models for different segmentation tasks
- Export analysis results to CSV/JSON
- Image preprocessing tools (crop, rotate, adjust)
- Comparison view for multiple images
- Undo/redo functionality

## Notes
- Prioritize code readability and maintainability
- Comment complex algorithms and non-obvious logic
- Keep the UI simple and intuitive for the initial version
- Build incrementally: start with basic features, then add ML capabilities
- Ensure cross-platform compatibility (Windows, macOS, Linux)
