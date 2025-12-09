"""Main application window for Image Analyzer."""

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from typing import Optional
import threading

import config
from utils.image_loader import load_image, validate_image_format
from utils.metrics import extract_image_metrics
from models.segmentation import SegmentationModel


class MainWindow:
    """Main application window for the Image Analyzer."""
    
    def __init__(self):
        """Initialize the main window."""
        self.root = ttk.Window(themename=config.THEME)
        self.root.title(config.APP_TITLE)
        self.root.geometry(f"{config.APP_WIDTH}x{config.APP_HEIGHT}")
        
        self.current_image: Optional[Image.Image] = None
        self.current_filepath: str = ""
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.segmentation_model: Optional[SegmentationModel] = None
        self.segmented_mask = None
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create and layout all widgets."""
        # Top toolbar
        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(side=TOP, fill=X)
        
        load_btn = ttk.Button(
            toolbar,
            text="Load Image",
            bootstyle=PRIMARY,
            command=self.load_image
        )
        load_btn.pack(side=LEFT, padx=5)
        
        self.ml_btn = ttk.Button(
            toolbar,
            text="Run ML Segmentation",
            bootstyle=SUCCESS,
            command=self.run_ml_segmentation,
            state=DISABLED
        )
        self.ml_btn.pack(side=LEFT, padx=5)
        
        # Main content area
        content = ttk.Frame(self.root)
        content.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Image display
        left_panel = ttk.Labelframe(content, text="Image Display", padding=10)
        left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 5))
        
        self.image_label = ttk.Label(left_panel, text="No image loaded", anchor=CENTER)
        self.image_label.pack(fill=BOTH, expand=True)
        
        # Right panel - Metrics
        right_panel = ttk.Labelframe(content, text="Image Metrics", padding=10)
        right_panel.pack(side=RIGHT, fill=BOTH, padx=(5, 0))
        
        self.metrics_text = ttk.Text(right_panel, width=30, height=20, wrap=WORD)
        self.metrics_text.pack(fill=BOTH, expand=True)
        self.metrics_text.config(state=DISABLED)
        
        # Status bar
        self.status_bar = ttk.Label(
            self.root,
            text="Ready",
            relief=SUNKEN,
            anchor=W,
            padding=(10, 5)
        )
        self.status_bar.pack(side=BOTTOM, fill=X)
        
    def load_image(self):
        """Open file dialog and load selected image."""
        filepath = filedialog.askopenfilename(filetypes=config.SUPPORTED_FORMATS)
        
        if not filepath:
            return
            
        if not validate_image_format(filepath):
            messagebox.showerror("Error", "Unsupported image format")
            return
            
        image = load_image(filepath)
        
        if image is None:
            messagebox.showerror("Error", "Failed to load image")
            return
            
        self.current_image = image
        self.current_filepath = filepath
        
        self._display_image()
        self._display_metrics()
        self.status_bar.config(text=f"Loaded: {filepath}")
        
        # Enable ML button when image is loaded
        self.ml_btn.config(state=NORMAL)
        
    def _display_image(self):
        """Display the current image in the image label."""
        if self.current_image is None:
            return
            
        # Calculate scaling to fit display area
        display_width = config.MAX_DISPLAY_WIDTH
        display_height = config.MAX_DISPLAY_HEIGHT
        
        img_width, img_height = self.current_image.size
        width_ratio = display_width / img_width
        height_ratio = display_height / img_height
        scale_ratio = min(width_ratio, height_ratio, 1.0)
        
        new_width = int(img_width * scale_ratio)
        new_height = int(img_height * scale_ratio)
        
        # Resize image
        display_image = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(display_image)
        self.image_label.config(image=self.photo_image, text="")
        
    def _display_metrics(self):
        """Display image metrics in the metrics panel."""
        if self.current_image is None:
            return
            
        metrics = extract_image_metrics(self.current_image, self.current_filepath)
        
        # Format metrics for display
        metrics_text = f"Resolution: {metrics['resolution']}\n"
        metrics_text += f"Channels: {metrics['channels']}\n"
        metrics_text += f"Color Space: {metrics['color_space']}\n"
        metrics_text += f"Bit Depth: {metrics['bit_depth']} bits\n"
        metrics_text += f"Format: {metrics['format']}\n"
        metrics_text += f"File Size: {metrics['file_size']}\n"
        
        self.metrics_text.config(state=NORMAL)
        self.metrics_text.delete(1.0, END)
        self.metrics_text.insert(1.0, metrics_text)
        self.metrics_text.config(state=DISABLED)
    
    def run_ml_segmentation(self):
        """Run ML segmentation on the current image."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        # Disable button during processing
        self.ml_btn.config(state=DISABLED)
        self.status_bar.config(text="Initializing ML model...")
        
        # Run in separate thread to avoid freezing UI
        thread = threading.Thread(target=self._run_segmentation_thread, daemon=True)
        thread.start()
    
    def _run_segmentation_thread(self):
        """Thread function to run segmentation."""
        try:
            # Step 1: Check system and load model
            if self.segmentation_model is None:
                self.segmentation_model = SegmentationModel()
            
            # Update UI - using after to run on main thread
            self.root.after(0, lambda: self.status_bar.config(text="Step 1: Checking system capabilities..."))
            
            success, message = self.segmentation_model.load_model()
            
            if not success:
                self.root.after(0, lambda: messagebox.showerror("Model Load Failed", message))
                self.root.after(0, lambda: self.ml_btn.config(state=NORMAL))
                self.root.after(0, lambda: self.status_bar.config(text="Ready"))
                return
            
            # Show system info
            self.root.after(0, lambda: messagebox.showinfo("System Check Complete", message))
            
            # Step 2: Run segmentation
            self.root.after(0, lambda: self.status_bar.config(text="Step 2: Running segmentation..."))
            
            mask = self.segmentation_model.segment_image(self.current_image)
            
            if mask is None:
                self.root.after(0, lambda: messagebox.showerror("Segmentation Failed", "Failed to segment image"))
                self.root.after(0, lambda: self.ml_btn.config(state=NORMAL))
                self.root.after(0, lambda: self.status_bar.config(text="Ready"))
                return
            
            self.segmented_mask = mask
            
            # Step 3: Display result
            self.root.after(0, lambda: self.status_bar.config(text="Step 3: Displaying results..."))
            
            overlay_image = self.segmentation_model.overlay_mask(self.current_image, mask, alpha=0.6)
            
            # Update display on main thread
            self.root.after(0, lambda: self._display_segmented_image(overlay_image))
            self.root.after(0, lambda: self.ml_btn.config(state=NORMAL))
            self.root.after(0, lambda: self.status_bar.config(text="Segmentation complete!"))
            
        except Exception as e:
            error_msg = f"Error during segmentation: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.ml_btn.config(state=NORMAL))
            self.root.after(0, lambda: self.status_bar.config(text="Ready"))
    
    def _display_segmented_image(self, image: Image.Image):
        """Display the segmented image."""
        # Calculate scaling to fit display area
        display_width = config.MAX_DISPLAY_WIDTH
        display_height = config.MAX_DISPLAY_HEIGHT
        
        img_width, img_height = image.size
        width_ratio = display_width / img_width
        height_ratio = display_height / img_height
        scale_ratio = min(width_ratio, height_ratio, 1.0)
        
        new_width = int(img_width * scale_ratio)
        new_height = int(img_height * scale_ratio)
        
        # Resize image
        display_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(display_image)
        self.image_label.config(image=self.photo_image, text="")
        
    def run(self):
        """Start the application main loop."""
        self.root.mainloop()

