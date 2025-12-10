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
        
        # Set minimum window size to prevent UI from breaking
        self.root.minsize(800, 600)
        
        self.current_image: Optional[Image.Image] = None
        self.current_filepath: str = ""
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.segmentation_model: Optional[SegmentationModel] = None
        self.segmented_mask = None
        self.display_mode: str = "original"  # "original" or "segmented"
        
        self._create_widgets()
        
        # Bind window resize event to update image display
        self.root.bind('<Configure>', self._on_window_resize)
        
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
        
        # Right panel - Metrics (fixed width to prevent disappearing)
        right_panel = ttk.Labelframe(content, text="Image Metrics", padding=10)
        right_panel.pack(side=RIGHT, fill=BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)  # Prevent shrinking below minimum
        right_panel.config(width=320)  # Fixed minimum width for readability
        
        self.metrics_text = ttk.Text(right_panel, width=35, height=8, wrap=WORD)
        self.metrics_text.pack(fill=X, pady=(0, 10))
        self.metrics_text.config(state=DISABLED)
        
        # Segmentation results panel
        seg_results_frame = ttk.Labelframe(right_panel, text="Segmentation Results", padding=10)
        seg_results_frame.pack(fill=BOTH, expand=True)
        
        # Overlay toggle checkbox
        overlay_control_frame = ttk.Frame(seg_results_frame)
        overlay_control_frame.pack(fill=X, pady=(0, 5))
        
        self.show_overlay_var = ttk.BooleanVar(value=True)
        self.overlay_checkbox = ttk.Checkbutton(
            overlay_control_frame,
            text="Show Overlay",
            variable=self.show_overlay_var,
            command=self._toggle_overlay,
            bootstyle="round-toggle"
        )
        self.overlay_checkbox.pack(side=LEFT)
        self.overlay_checkbox.config(state=DISABLED)  # Initially disabled until segmentation runs
        
        # Create scrollable frame for segmentation results
        self.seg_canvas = ttk.Canvas(seg_results_frame)
        seg_scrollbar = ttk.Scrollbar(seg_results_frame, orient=VERTICAL, command=self.seg_canvas.yview)
        self.seg_results_frame_inner = ttk.Frame(self.seg_canvas)
        
        self.seg_results_frame_inner.bind(
            "<Configure>",
            lambda e: self.seg_canvas.configure(scrollregion=self.seg_canvas.bbox("all"))
        )
        
        self.seg_canvas.create_window((0, 0), window=self.seg_results_frame_inner, anchor="nw")
        self.seg_canvas.configure(yscrollcommand=seg_scrollbar.set)
        
        self.seg_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        seg_scrollbar.pack(side=RIGHT, fill=Y)
        
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
    
    def _toggle_overlay(self):
        """Toggle between original image and segmented overlay."""
        if self.show_overlay_var.get():
            self.display_mode = "segmented"
        else:
            self.display_mode = "original"
        self._display_image()
        
    def _display_image(self):
        """Display the current image in the image label."""
        if self.current_image is None:
            return
        
        # Determine which image to display
        if self.display_mode == "segmented" and self.segmented_mask is not None:
            # Display segmented overlay
            image_to_display = self.segmentation_model.overlay_mask(self.current_image, self.segmented_mask, alpha=0.6)
        else:
            # Display original image
            image_to_display = self.current_image
        
        # Get actual available space for image display
        self.image_label.update_idletasks()
        display_width = max(self.image_label.winfo_width() - 20, 400)  # Leave some padding
        display_height = max(self.image_label.winfo_height() - 20, 400)
        
        img_width, img_height = image_to_display.size
        width_ratio = display_width / img_width
        height_ratio = display_height / img_height
        scale_ratio = min(width_ratio, height_ratio, 1.0)
        
        new_width = int(img_width * scale_ratio)
        new_height = int(img_height * scale_ratio)
        
        # Resize image
        display_image = image_to_display.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
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
    
    def _display_segmentation_results(self, stats: list):
        """
        Display segmentation results with color swatches, labels, and percentages.
        
        Args:
            stats: List of segmentation statistics from model
        """
        # Clear previous results
        for widget in self.seg_results_frame_inner.winfo_children():
            widget.destroy()
        
        if not stats:
            no_results = ttk.Label(
                self.seg_results_frame_inner,
                text="No segmentation results",
                anchor=CENTER
            )
            no_results.pack(pady=20)
            return
        
        # Display each class with color swatch
        for idx, stat in enumerate(stats):
            result_frame = ttk.Frame(self.seg_results_frame_inner)
            result_frame.pack(fill=X, pady=2, padx=5)
            
            # Color swatch (20x20 pixel canvas)
            color_canvas = ttk.Canvas(result_frame, width=20, height=20)
            color_canvas.pack(side=LEFT, padx=(0, 10))
            
            # Use actual color if percentage >= 0.1%, otherwise show empty black box
            if stat['percentage'] >= 0.1:
                color_hex = '#{:02x}{:02x}{:02x}'.format(*stat['color'])
                color_canvas.create_rectangle(0, 0, 20, 20, fill=color_hex, outline="black")
            else:
                # Empty box with just black outline for classes not detected
                color_canvas.create_rectangle(0, 0, 20, 20, fill="", outline="black")
            
            # Class name and percentage
            label_text = f"{stat['name']}: {stat['percentage']:.1f}%"
            result_label = ttk.Label(result_frame, text=label_text)
            result_label.pack(side=LEFT, fill=X, expand=True)
    
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
            
            # Step 3: Calculate statistics
            self.root.after(0, lambda: self.status_bar.config(text="Step 3: Calculating statistics..."))
            
            stats = self.segmentation_model.get_segmentation_stats(mask)
            
            # Step 4: Display results
            self.root.after(0, lambda: self.status_bar.config(text="Step 4: Displaying results..."))
            
            # Switch to segmented display mode
            self.display_mode = "segmented"
            
            # Update display on main thread
            self.root.after(0, lambda: self._display_image())
            self.root.after(0, lambda: self._display_segmentation_results(stats))
            self.root.after(0, lambda: self.ml_btn.config(state=NORMAL))
            self.root.after(0, lambda: self.overlay_checkbox.config(state=NORMAL))
            self.root.after(0, lambda: self.status_bar.config(text="Segmentation complete!"))
            
        except Exception as e:
            error_msg = f"Error during segmentation: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.ml_btn.config(state=NORMAL))
            self.root.after(0, lambda: self.status_bar.config(text="Ready"))
    
    def _on_window_resize(self, event):
        """Handle window resize events to update image display."""
        # Only respond to resize events from the root window
        if event.widget == self.root and self.current_image is not None:
            # Use after to debounce resize events
            if hasattr(self, '_resize_id'):
                self.root.after_cancel(self._resize_id)
            self._resize_id = self.root.after(100, self._display_image)
        
    def run(self):
        """Start the application main loop."""
        self.root.mainloop()

