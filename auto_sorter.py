import os
import sys
import shutil
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import config
from cnn_classifier import CNNClassifier

class AutoSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Sorter - CNN Powered")
        self.root.geometry("600x480")
        self.root.configure(bg="#1e1e1e")
        
        self.unsorted_dir = ""
        self.images = []
        self.processed_count = 0
        self.total_count = 0
        self.start_time = 0
        self.is_sorting = False
        
        self.classifier = None
        
        # Load default output dir from config, but don't hardcode it
        dir_paths = getattr(config, "TRAINING_DIRECTORIES", [])
        self.gold_standard_dir = dir_paths[0] if dir_paths else ""
        
        self.setup_ui()
        
        # Load config and model in background
        threading.Thread(target=self.init_backend, daemon=True).start()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TProgressbar", thickness=30, background="#56b6c2")
        
        main_frame = tk.Frame(self.root, bg="#1e1e1e", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="AI Automated Image Sorter", font=("Arial", 18, "bold"), bg="#1e1e1e", fg="white").pack(pady=10)
        
        self.lbl_status = tk.Label(main_frame, text="Loading CNN Model...", font=("Arial", 12), bg="#1e1e1e", fg="#abb2bf")
        self.lbl_status.pack(pady=5)
        
        self.btn_select = tk.Button(main_frame, text="1. Select Unsorted Input Folder", font=("Arial", 11), bg="#3d3d3d", fg="white", command=self.select_input_folder, state=tk.DISABLED)
        self.btn_select.pack(pady=10)
        
        self.lbl_folder = tk.Label(main_frame, text="No input folder selected", font=("Arial", 9), bg="#1e1e1e", fg="#98c379")
        self.lbl_folder.pack(pady=2)
        
        self.btn_select_out = tk.Button(main_frame, text="2. Select Destination Folder", font=("Arial", 11), bg="#3d3d3d", fg="white", command=self.select_output_folder, state=tk.DISABLED)
        self.btn_select_out.pack(pady=10)
        
        self.lbl_folder_out = tk.Label(main_frame, text=self.gold_standard_dir if self.gold_standard_dir else "No output folder selected", font=("Arial", 9), bg="#1e1e1e", fg="#e5c07b")
        self.lbl_folder_out.pack(pady=2)
        
        self.btn_start = tk.Button(main_frame, text="Start Sorting ⚡", font=("Arial", 14, "bold"), bg="#56b6c2", fg="black", command=self.start_sorting, state=tk.DISABLED)
        self.btn_start.pack(pady=15)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, style="TProgressbar", variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=10)
        
        self.lbl_stats = tk.Label(main_frame, text="", font=("Arial", 12), bg="#1e1e1e", fg="white")
        self.lbl_stats.pack(pady=5)

    def init_backend(self):
        try:
            self.classifier = CNNClassifier()
            if self.classifier.model is None:
                self.root.after(0, lambda: self.lbl_status.config(text="Error: Model missing. Check train_cnn.py", fg="#e06c75"))
                return
                
            # Enable UI
            self.root.after(0, self.backend_ready)
        except Exception as e:
            self.root.after(0, lambda: self.lbl_status.config(text=f"Initialization Error: {e}", fg="#e06c75"))

    def backend_ready(self):
        self.lbl_status.config(text=f"Model Ready ({len(self.classifier.classes)} classes loaded)", fg="#98c379")
        self.btn_select.config(state=tk.NORMAL)
        self.btn_select_out.config(state=tk.NORMAL)
        self.check_ready_to_start()

    def select_input_folder(self):
        folder = filedialog.askdirectory(title="Select the folder containing UNSORTED images")
        if not folder: return
        
        self.unsorted_dir = folder
        self.lbl_folder.config(text=self.unsorted_dir)
        
        self.images = []
        for root_dir, dirs, files in os.walk(self.unsorted_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    self.images.append(os.path.join(root_dir, f))
                    
        self.total_count = len(self.images)
        self.check_ready_to_start()
        
    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select DESTINATION folder")
        if not folder: return
        
        self.gold_standard_dir = folder
        self.lbl_folder_out.config(text=self.gold_standard_dir)
        self.check_ready_to_start()
        
    def check_ready_to_start(self):
        if self.total_count > 0 and self.gold_standard_dir:
            self.lbl_stats.config(text=f"Ready to sort {self.total_count} images.")
            self.btn_start.config(state=tk.NORMAL)
        elif self.total_count == 0 and self.unsorted_dir:
            self.lbl_stats.config(text="No images found in input folder.")
            self.btn_start.config(state=tk.DISABLED)
        else:
            self.btn_start.config(state=tk.DISABLED)

    def start_sorting(self):
        if self.is_sorting: return
        
        self.is_sorting = True
        self.btn_select.config(state=tk.DISABLED)
        self.btn_select_out.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        self.processed_count = 0
        self.progress_var.set(0)
        self.start_time = time.time()
        
        self.lbl_status.config(text=f"Sorting into {self.gold_standard_dir}...", fg="#e5c07b")
        
        # Start sort thread
        threading.Thread(target=self.sort_thread, daemon=True).start()
        
        # Start GUI updater
        self.root.after(100, self.update_gui)

    def sort_thread(self):
        # Create output directories on the fly
        os.makedirs(self.gold_standard_dir, exist_ok=True)
        for cls in self.classifier.classes:
            os.makedirs(os.path.join(self.gold_standard_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(self.gold_standard_dir, "Amorphous"), exist_ok=True)
        os.makedirs(os.path.join(self.gold_standard_dir, "Review"), exist_ok=True)

        reset_every = max(1, int(getattr(config, "MEMORY_RESET_EVERY_N_IMAGES", 500)))
        
        for img_path in self.images:
            img_name = os.path.basename(img_path)
            predicted_class = self.classifier.classify(img_path)
            
            if predicted_class != "ignore":
                target_dir = os.path.join(self.gold_standard_dir, predicted_class)
                new_path = os.path.join(target_dir, img_name)
                
                base, ext = os.path.splitext(img_name)
                counter = 1
                while os.path.exists(new_path):
                    new_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
                    counter += 1
                    
                try:
                    shutil.move(img_path, new_path)
                except Exception as e:
                    print(f"Failed to move {img_name}: {e}")
                    
            self.processed_count += 1

            # Periodic memory flush — prevents OOM on ~1M image runs
            if self.processed_count % reset_every == 0:
                self.classifier.release_memory()
                print(
                    f"[memory] Reset after {self.processed_count}/{self.total_count} images"
                )

        # Final flush when the run finishes
        self.classifier.release_memory()
        self.is_sorting = False

    def update_gui(self):
        if self.total_count > 0:
            percent = (self.processed_count / self.total_count) * 100
            self.progress_var.set(percent)
            
            elapsed = time.time() - self.start_time
            if self.processed_count > 0:
                speed = self.processed_count / elapsed
                remaining_items = self.total_count - self.processed_count
                eta_seconds = remaining_items / speed
                
                m, s = divmod(int(eta_seconds), 60)
                h, m = divmod(m, 60)
                eta_str = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s"
                
                self.lbl_stats.config(text=f"Sorted: {self.processed_count} / {self.total_count} ({percent:.1f}%) | ETA: {eta_str}")
            
        if self.is_sorting:
            self.root.after(200, self.update_gui)
        else:
            # Finished
            self.progress_var.set(100)
            self.lbl_status.config(text="Sorting Complete!", fg="#98c379")
            self.btn_select.config(state=tk.NORMAL)
            self.btn_select_out.config(state=tk.NORMAL)
            messagebox.showinfo("Done", f"Successfully sorted {self.processed_count} images!")

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoSorterApp(root)
    root.mainloop()
