"""
Image Analyzer Application

A Python-based image analysis tool with a modern GUI for loading,
viewing, and analyzing images. Features include metadata extraction
and ML-based segmentation capabilities.
"""

from gui.main_window import MainWindow


def main():
    """Entry point for the Image Analyzer application."""
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()
