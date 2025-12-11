"""Configuration settings for the Image Analyzer application."""

# Application settings
APP_TITLE = "Image Analyzer"
APP_WIDTH = 1200
APP_HEIGHT = 800
APP_MIN_WIDTH = 800
APP_MIN_HEIGHT = 600
THEME = "darkly"

# Supported image formats
SUPPORTED_FORMATS = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.gif *.webp"),
    ("All files", "*.*")
]

# Image display settings
MAX_DISPLAY_WIDTH = 800
MAX_DISPLAY_HEIGHT = 600
