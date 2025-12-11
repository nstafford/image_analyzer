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
from utils.system_checker import get_system_info, format_system_info
from models.segmentation import SegmentationModel


class MainWindow:
    """Main application window for the Image Analyzer."""
    
    def __init__(self):
        """Initialize the main window."""
        self.root = ttk.Window(themename=config.THEME)
        self.root.title(config.APP_TITLE)
        self.root.geometry(f"{config.APP_WIDTH}x{config.APP_HEIGHT}")
        
        # Set minimum window size to prevent UI from breaking
        self.root.minsize(config.APP_MIN_WIDTH, config.APP_MIN_HEIGHT)
        
        self.current_image: Optional[Image.Image] = None
        self.current_filepath: str = ""
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.segmentation_model: Optional[SegmentationModel] = None
        self.segmented_mask = None
        self.display_mode: str = "original"  # "original" or "segmented"
        self.model_loaded: bool = False
        self.loaded_model_name: str = ""
        self.system_info = None
        
        self._create_widgets()
        self._check_system_on_startup()
        
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
        
        self.load_model_btn = ttk.Button(
            toolbar,
            text="Load ML Model",
            bootstyle=INFO,
            command=self.load_ml_model
        )
        self.load_model_btn.pack(side=LEFT, padx=5)
        
        self.loaded_model_label = ttk.Label(
            toolbar,
            text="No model loaded",
            bootstyle=SECONDARY
        )
        self.loaded_model_label.pack(side=LEFT, padx=5)
        
        self.apply_model_btn = ttk.Button(
            toolbar,
            text="Apply Model to Image",
            bootstyle=SUCCESS,
            command=self.apply_model_to_image,
            state=DISABLED
        )
        self.apply_model_btn.pack(side=LEFT, padx=5)
        
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
        
        self.metrics_text = ttk.Text(right_panel, width=35, height=6, wrap=WORD)
        self.metrics_text.pack(fill=X, pady=(0, 10))
        self.metrics_text.config(state=DISABLED)
        
        # System info panel
        system_info_frame = ttk.Labelframe(right_panel, text="System Information", padding=10)
        system_info_frame.pack(fill=X, pady=(0, 10))
        
        self.system_info_text = ttk.Text(system_info_frame, width=35, height=6, wrap=WORD)
        self.system_info_text.pack(fill=X)
        self.system_info_text.config(state=DISABLED)
        
        # Model info panel
        model_info_frame = ttk.Labelframe(right_panel, text="Model Information", padding=10)
        model_info_frame.pack(fill=X, pady=(0, 10))
        
        self.model_info_text = ttk.Text(model_info_frame, width=35, height=5, wrap=WORD)
        self.model_info_text.pack(fill=X)
        self.model_info_text.config(state=NORMAL)
        self.model_info_text.insert(1.0, "None")
        self.model_info_text.config(state=DISABLED)
        
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
        
        # Reset segmentation state for new image
        self.segmented_mask = None
        self.display_mode = "original"
        self.show_overlay_var.set(True)
        self.overlay_checkbox.config(state=DISABLED)
        
        # Clear segmentation results
        for widget in self.seg_results_frame_inner.winfo_children():
            widget.destroy()
        
        self._display_image()
        self._display_metrics()
        self.status_bar.config(text=f"Loaded: {filepath}")
        
        # Enable Apply Model button if model is loaded
        if self.model_loaded:
            self.apply_model_btn.config(state=NORMAL)
    
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
    
    def _display_model_info(self):
        """Display model information in the model info panel."""
        if self.segmentation_model is None or not self.model_loaded:
            # Show "None" when no model is loaded
            self.model_info_text.config(state=NORMAL)
            self.model_info_text.delete(1.0, END)
            self.model_info_text.insert(1.0, "None")
            self.model_info_text.config(state=DISABLED)
            return
        
        model_info = self.segmentation_model.get_model_info()
        
        # Format model info for display
        info_text = f"Model: {model_info['name']}\n"
        info_text += f"Parameters: {model_info['num_parameters']:,}\n"
        info_text += f"Size: {model_info['size_mb']:.2f} MB\n"
        info_text += f"Weights: {model_info['weights']}\n"
        info_text += f"Device: {model_info['device']}\n"
        
        self.model_info_text.config(state=NORMAL)
        self.model_info_text.delete(1.0, END)
        self.model_info_text.insert(1.0, info_text)
        self.model_info_text.config(state=DISABLED)
    
    def _check_system_on_startup(self):
        """Check system capabilities at startup and display info in background thread."""
        # Show loading message immediately
        self.system_info_text.config(state=NORMAL)
        self.system_info_text.delete(1.0, END)
        self.system_info_text.insert(1.0, "Checking system capabilities...")
        self.system_info_text.config(state=DISABLED)
        
        def check_thread():
            try:
                self.system_info = get_system_info()
                formatted_info = format_system_info(self.system_info)
                
                self.root.after(0, lambda: self.system_info_text.config(state=NORMAL))
                self.root.after(0, lambda: self.system_info_text.delete(1.0, END))
                self.root.after(0, lambda: self.system_info_text.insert(1.0, formatted_info))
                self.root.after(0, lambda: self.system_info_text.config(state=DISABLED))
                self.root.after(0, lambda: self.status_bar.config(text="System check complete"))
            except Exception as e:
                error_msg = f"Error checking system: {str(e)}"
                self.root.after(0, lambda: self.system_info_text.config(state=NORMAL))
                self.root.after(0, lambda: self.system_info_text.delete(1.0, END))
                self.root.after(0, lambda: self.system_info_text.insert(1.0, error_msg))
                self.root.after(0, lambda: self.system_info_text.config(state=DISABLED))
        
        threading.Thread(target=check_thread, daemon=True).start()
    
    def load_ml_model(self):
        """Show model selection dialog and load chosen model."""
        # For now, only one model available
        available_models = [
            {"name": "DeepLabV3 (MobileNetV3)", "id": "deeplabv3_mobilenet_v3"}
        ]
        
        # Create simple selection dialog
        dialog = ttk.Toplevel(self.root)
        dialog.title("Select ML Model")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(
            dialog,
            text="Select a segmentation model to load:",
            padding=10
        ).pack()
        
        selected_model = ttk.StringVar(value=available_models[0]["id"])
        
        for model in available_models:
            ttk.Radiobutton(
                dialog,
                text=model["name"],
                variable=selected_model,
                value=model["id"]
            ).pack(anchor=W, padx=20, pady=5)
        
        def on_load():
            model_id = selected_model.get()
            dialog.destroy()
            self._load_model_thread(model_id)
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(side=BOTTOM, pady=10)
        
        ttk.Button(
            button_frame,
            text="Load Model",
            bootstyle=SUCCESS,
            command=on_load
        ).pack(side=LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            bootstyle=SECONDARY,
            command=dialog.destroy
        ).pack(side=LEFT, padx=5)
    
    def _load_model_thread(self, model_id: str):
        """Load the selected model in a background thread."""
        self.load_model_btn.config(state=DISABLED)
        self.status_bar.config(text="Loading ML model...")
        
        def load_thread():
            try:
                self.segmentation_model = SegmentationModel()
                success, message = self.segmentation_model.load_model()
                
                if success:
                    self.model_loaded = True
                    # Get model name from ID
                    if model_id == "deeplabv3_mobilenet_v3":
                        self.loaded_model_name = "DeepLabV3"
                    
                    self.root.after(0, lambda: self.loaded_model_label.config(
                        text=f"Loaded: {self.loaded_model_name}",
                        bootstyle=SUCCESS
                    ))
                    self.root.after(0, lambda: self._display_model_info())
                    # self.root.after(0, lambda: messagebox.showinfo("Model Loaded", message))
                    self.root.after(0, lambda: self.status_bar.config(text=f"Model loaded: {self.loaded_model_name}"))
                    
                    # Enable Apply Model button if image is loaded
                    if self.current_image is not None:
                        self.root.after(0, lambda: self.apply_model_btn.config(state=NORMAL))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Model Load Failed", message))
                    self.root.after(0, lambda: self.status_bar.config(text="Ready"))
                
                self.root.after(0, lambda: self.load_model_btn.config(state=NORMAL))
                
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                self.root.after(0, lambda: self.load_model_btn.config(state=NORMAL))
                self.root.after(0, lambda: self.status_bar.config(text="Ready"))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def apply_model_to_image(self):
        """Apply the loaded model to the current image."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        if not self.model_loaded:
            messagebox.showwarning("No Model", "Please load a model first")
            return
        
        # Disable button during processing
        self.apply_model_btn.config(state=DISABLED)
        self.status_bar.config(text="Applying model to image...")
        
        # Run in separate thread to avoid freezing UI
        thread = threading.Thread(target=self._apply_model_thread, daemon=True)
        thread.start()
    
    def _apply_model_thread(self):
        """Thread function to apply model to image."""
    def _apply_model_thread(self):
        """Thread function to apply model to image."""
        try:
            # Run segmentation
            self.root.after(0, lambda: self.status_bar.config(text="Running segmentation..."))
            
            mask = self.segmentation_model.segment_image(self.current_image)
            
            if mask is None:
                self.root.after(0, lambda: messagebox.showerror("Segmentation Failed", "Failed to segment image"))
                self.root.after(0, lambda: self.apply_model_btn.config(state=NORMAL))
                self.root.after(0, lambda: self.status_bar.config(text="Ready"))
                return
            
            self.segmented_mask = mask
            
            # Calculate statistics
            self.root.after(0, lambda: self.status_bar.config(text="Calculating statistics..."))
            
            stats = self.segmentation_model.get_segmentation_stats(mask)
            
            # Display results
            self.root.after(0, lambda: self.status_bar.config(text="Displaying results..."))
            
            # Switch to segmented display mode
            self.display_mode = "segmented"
            
            # Update display on main thread
            self.root.after(0, lambda: self._display_image())
            self.root.after(0, lambda: self._display_segmentation_results(stats))
            self.root.after(0, lambda: self.apply_model_btn.config(state=NORMAL))
            self.root.after(0, lambda: self.overlay_checkbox.config(state=NORMAL))
            self.root.after(0, lambda: self.status_bar.config(text="Segmentation complete!"))
            
        except Exception as e:
            error_msg = f"Error during segmentation: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.apply_model_btn.config(state=NORMAL))
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

