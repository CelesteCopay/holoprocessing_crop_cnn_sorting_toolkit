import os
import sys
import shutil
import json
from collections import defaultdict

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import config


def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


PROGRESS_FILE = os.path.join(_app_dir(), "fast_sorter_progress.json")
BASE_CELL = 120
ZOOM_MIN = 0.25
ZOOM_MAX = 3.0
GALLERY_PAD = 4
GALLERY_ROW_BUFFER = 4
GALLERY_LOAD_PER_TICK = 8


class FastSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lightning Fast Image Sorter ⚡")
        self.root.geometry("1100x800")
        self.root.configure(bg="#1e1e1e")

        # Core state
        self.current_dir = ""
        self.image_files = []
        self.current_index = 0
        self.categories = []
        self.undo_stack = []
        self.progress_data = self.load_progress()

        # View / zoom
        self.view_mode = "slideshow"
        self.zoom_level = 1.0
        self._resize_job = None

        # Virtual gallery state
        self._gallery_layout = None
        self._gallery_items = {}  # index -> {border_id, image_id, photo}
        self._gallery_pending = []
        self._gallery_load_job = None
        self._gallery_sync_job = None
        self._gallery_cached_width = -1

        # Session / pass tracking
        self.passed_paths = set()
        self.seen_paths = set()
        self.session_passed = 0
        self.session_moved = defaultdict(int)
        self.session_total_at_load = 0
        self.selected_indices = set()
        self._selection_anchor = 0
        self._gallery_queue_count = -1
        self.all_images_at_load = []
        self.new_paths_at_load = set()
        self.show_new_only = False

        dir_paths = getattr(config, "TRAINING_DIRECTORIES", [])
        self.gold_standard_dir = ""
        for path in dir_paths:
            if path and os.path.isdir(path):
                self.gold_standard_dir = path
                break

        self.load_categories()
        self.setup_ui()
        self.setup_bindings()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ progress

    def load_progress(self):
        try:
            if os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading progress: {e}")
        return {}

    def _folder_progress_entry(self):
        entry = self.progress_data.get(self.current_dir, {})
        if isinstance(entry, int):
            return {"index": entry, "passed": [], "seen": []}
        return {
            "index": entry.get("index", 0),
            "passed": entry.get("passed", []),
            "seen": entry.get("seen", []),
        }

    def save_progress(self, commit_seen=False):
        if not self.current_dir:
            return
        if commit_seen:
            self.seen_paths |= set(self.all_images_at_load)
        self.progress_data[self.current_dir] = {
            "index": self.current_index,
            "passed": sorted(self.passed_paths),
            "seen": sorted(self.seen_paths),
        }
        try:
            with open(PROGRESS_FILE, "w") as f:
                json.dump(self.progress_data, f)
        except Exception as e:
            print(f"Error saving progress: {e}")

    def on_close(self):
        self.save_progress(commit_seen=True)
        self.root.destroy()

    def load_categories(self):
        """Categories = subfolder names under the selected destination folder."""
        category_set = set()
        if self.gold_standard_dir and os.path.isdir(self.gold_standard_dir):
            for name in os.listdir(self.gold_standard_dir):
                full_path = os.path.join(self.gold_standard_dir, name)
                if os.path.isdir(full_path):
                    category_set.add(name)

        self.categories = sorted(category_set)

    # ------------------------------------------------------------------ UI setup

    def setup_ui(self):
        self.top_frame = tk.Frame(self.root, bg="#2d2d2d", pady=10)
        self.top_frame.pack(fill=tk.X)

        self.btn_frame = tk.Frame(self.top_frame, bg="#2d2d2d")
        self.btn_frame.pack()

        self.btn_input = tk.Button(
            self.btn_frame,
            text="1. Select Input Folder",
            font=("Arial", 11),
            bg="#3d3d3d",
            fg="white",
            command=self.select_input_folder,
        )
        self.btn_input.grid(row=0, column=0, padx=10)

        self.btn_dest = tk.Button(
            self.btn_frame,
            text="2. Select Destination Folder",
            font=("Arial", 11),
            bg="#3d3d3d",
            fg="white",
            command=self.select_output_folder,
        )
        self.btn_dest.grid(row=0, column=1, padx=10)

        self.controls_frame = tk.Frame(self.top_frame, bg="#2d2d2d")
        self.controls_frame.pack(pady=5)

        self.mode_var = tk.StringVar(value="slideshow")
        tk.Radiobutton(
            self.controls_frame,
            text="Slideshow",
            variable=self.mode_var,
            value="slideshow",
            bg="#2d2d2d",
            fg="white",
            selectcolor="#3d3d3d",
            activebackground="#2d2d2d",
            activeforeground="white",
            command=self.switch_view_mode,
        ).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(
            self.controls_frame,
            text="Gallery",
            variable=self.mode_var,
            value="gallery",
            bg="#2d2d2d",
            fg="white",
            selectcolor="#3d3d3d",
            activebackground="#2d2d2d",
            activeforeground="white",
            command=self.switch_view_mode,
        ).pack(side=tk.LEFT, padx=5)

        tk.Label(
            self.controls_frame,
            text="Zoom:",
            bg="#2d2d2d",
            fg="#abb2bf",
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=(15, 2))

        self.zoom_var = tk.DoubleVar(value=1.0)
        self.zoom_scale = tk.Scale(
            self.controls_frame,
            from_=ZOOM_MIN,
            to=ZOOM_MAX,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.zoom_var,
            length=160,
            bg="#2d2d2d",
            fg="white",
            troughcolor="#3d3d3d",
            highlightthickness=0,
            command=self._on_zoom_change,
        )
        self.zoom_scale.pack(side=tk.LEFT, padx=2)

        self.lbl_zoom = tk.Label(
            self.controls_frame,
            text="100%",
            bg="#2d2d2d",
            fg="#abb2bf",
            font=("Arial", 10),
            width=5,
        )
        self.lbl_zoom.pack(side=tk.LEFT, padx=2)

        self.btn_mark_reviewed = tk.Button(
            self.controls_frame,
            text="Mark Reviewed (M)",
            font=("Arial", 10),
            bg="#3d3d3d",
            fg="white",
            command=self.mark_selected_reviewed,
        )
        self.btn_mark_reviewed.pack(side=tk.LEFT, padx=(15, 5))

        self.btn_mark_all = tk.Button(
            self.controls_frame,
            text="Mark All Reviewed (Ctrl+M)",
            font=("Arial", 10),
            bg="#3d3d3d",
            fg="white",
            command=self.mark_all_reviewed,
        )
        self.btn_mark_all.pack(side=tk.LEFT, padx=5)

        self.new_only_var = tk.BooleanVar(value=False)
        self.chk_new_only = tk.Checkbutton(
            self.controls_frame,
            text="New only (N)",
            variable=self.new_only_var,
            bg="#2d2d2d",
            fg="#e5c07b",
            selectcolor="#3d3d3d",
            activebackground="#2d2d2d",
            activeforeground="#e5c07b",
            font=("Arial", 10, "bold"),
            command=self.toggle_new_only,
        )
        self.chk_new_only.pack(side=tk.LEFT, padx=(15, 5))

        self.lbl_frame = tk.Frame(self.top_frame, bg="#2d2d2d")
        self.lbl_frame.pack(pady=5)

        self.lbl_input_disp = tk.Label(
            self.lbl_frame,
            text="Input: None",
            bg="#2d2d2d",
            fg="#98c379",
            font=("Arial", 10),
        )
        self.lbl_input_disp.grid(row=0, column=0, padx=20)

        self.lbl_dest_disp = tk.Label(
            self.lbl_frame,
            text=f"Dest: {self.gold_standard_dir}",
            bg="#2d2d2d",
            fg="#e5c07b",
            font=("Arial", 10),
        )
        self.lbl_dest_disp.grid(row=0, column=1, padx=20)

        self.lbl_info = tk.Label(
            self.top_frame,
            text="Select an input folder to begin...",
            bg="#2d2d2d",
            fg="white",
            font=("Arial", 14, "bold"),
        )
        self.lbl_info.pack(pady=5)

        self.lbl_stats = tk.Label(
            self.top_frame,
            text="",
            bg="#2d2d2d",
            fg="#abb2bf",
            font=("Arial", 10),
        )
        self.lbl_stats.pack(pady=2)

        self.legend_frame = tk.Frame(self.root, bg="#1e1e1e", pady=5)
        self.legend_frame.pack(fill=tk.X)

        self.update_legend()

        # Slideshow pane (scrollable canvas)
        self.slideshow_container = tk.Frame(self.root, bg="#1e1e1e")
        self.slideshow_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.slideshow_vsb = tk.Scrollbar(
            self.slideshow_container, orient=tk.VERTICAL
        )
        self.slideshow_hsb = tk.Scrollbar(
            self.slideshow_container, orient=tk.HORIZONTAL
        )
        self.canvas = tk.Canvas(
            self.slideshow_container,
            bg="#000000",
            highlightthickness=0,
            xscrollcommand=self.slideshow_hsb.set,
            yscrollcommand=self.slideshow_vsb.set,
        )
        self.slideshow_vsb.config(command=self.canvas.yview)
        self.slideshow_hsb.config(command=self.canvas.xview)

        self.slideshow_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.slideshow_hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Gallery pane (hidden initially)
        self.gallery_container = tk.Frame(self.root, bg="#1e1e1e")

        self.gallery_vsb = tk.Scrollbar(
            self.gallery_container, orient=tk.VERTICAL
        )
        self.gallery_canvas = tk.Canvas(
            self.gallery_container,
            bg="#000000",
            highlightthickness=0,
            yscrollcommand=self._gallery_on_yscroll,
        )
        self.gallery_vsb.config(command=self._gallery_yview)
        self.gallery_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.gallery_canvas.bind("<Configure>", self._on_gallery_canvas_configure)
        self.gallery_canvas.bind("<Button-1>", self._on_gallery_canvas_click)
        self.gallery_canvas.bind(
            "<Double-Button-1>", self._on_gallery_canvas_double_click
        )
        self.canvas.bind("<Button-1>", self._on_slideshow_click)
        self.canvas.bind("<Double-Button-1>", self._on_slideshow_click)

        self.bottom_frame = tk.Frame(self.root, bg="#2d2d2d")
        self.bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.lbl_help = tk.Label(
            self.bottom_frame,
            text="",
            bg="#2d2d2d",
            fg="#abb2bf",
            font=("Arial", 10),
            pady=5,
        )
        self.lbl_help.pack()
        self._update_help_text()

    def _on_gallery_canvas_configure(self, event):
        if self.view_mode != "gallery":
            return
        if event.width == self._gallery_cached_width:
            return
        self._gallery_cached_width = event.width
        self._gallery_schedule_sync(full_relayout=True)

    def _gallery_yview(self, *args):
        self.gallery_canvas.yview(*args)
        self._gallery_schedule_sync()

    def _gallery_on_yscroll(self, first, last):
        self.gallery_vsb.set(first, last)
        self._gallery_schedule_sync()

    def _gallery_schedule_sync(self, full_relayout=False):
        if self.view_mode != "gallery":
            return
        if self._gallery_sync_job:
            self.root.after_cancel(self._gallery_sync_job)

        def run():
            self._gallery_sync_job = None
            if full_relayout or self._gallery_layout is None:
                self._gallery_relayout()
            else:
                self._gallery_sync_visible()

        self._gallery_sync_job = self.root.after(30, run)

    def _update_help_text(self):
        if self.view_mode == "slideshow":
            self.lbl_help.config(
                text=(
                    "Left/Right: Navigate | Click / Esc: Back to gallery | "
                    "1-0: Cat 1-10 | Shift+1-0: Cat 11-20 | Ctrl+1-0: Cat 21-30 | "
                    "F: Favorite | M: Mark reviewed | Ctrl+M: Mark all | "
                    "N: New only | Ctrl+Z: Undo | +/-: Zoom"
                )
            )
        else:
            self.lbl_help.config(
                text=(
                    "Click: Select | Double-click: Open slideshow | "
                    "Ctrl+Click: Toggle | Shift+Click: Range/rows | "
                    "1-0/F: Sort selection | M / Ctrl+M: Reviewed | "
                    "N: New only | Esc: Clear selection | Ctrl+Z: Undo | +/-: Zoom"
                )
            )

    def switch_view_mode(self, refresh=True):
        self.view_mode = self.mode_var.get()
        if self.view_mode == "slideshow":
            self.gallery_container.pack_forget()
            self.slideshow_container.pack(
                fill=tk.BOTH, expand=True, padx=20, pady=10
            )
        else:
            self.slideshow_container.pack_forget()
            self.gallery_container.pack(
                fill=tk.BOTH, expand=True, padx=20, pady=10
            )
            self._gallery_cached_width = -1
        self._update_help_text()
        if refresh:
            self.refresh_view(
                scroll_gallery_to_selection=(self.view_mode == "gallery"),
                gallery_force=(self.view_mode == "gallery"),
            )

    def enter_slideshow_at(self, index):
        if not self.image_files:
            return
        index = max(0, min(index, len(self.image_files) - 1))
        self.current_index = index
        self.selected_indices = {index}
        self._selection_anchor = index
        self.mode_var.set("slideshow")
        self.switch_view_mode(refresh=True)

    def return_to_gallery(self):
        if self.view_mode != "slideshow":
            return
        self.mode_var.set("gallery")
        self.switch_view_mode(refresh=True)

    def on_escape(self):
        if self.view_mode == "slideshow":
            self.return_to_gallery()
        else:
            self.clear_selection()

    def toggle_new_only(self):
        self.show_new_only = bool(self.new_only_var.get())
        if not self.current_dir:
            return
        self._rebuild_queue_from_filters(preserve_selection=True)

    def update_legend(self):
        for widget in self.legend_frame.winfo_children():
            widget.destroy()

        self.category_labels = {}

        row1 = tk.Frame(self.legend_frame, bg="#1e1e1e")
        row1.pack()
        row2 = tk.Frame(self.legend_frame, bg="#1e1e1e")
        row2.pack()
        row3 = tk.Frame(self.legend_frame, bg="#1e1e1e")
        row3.pack()

        for i, cat in enumerate(self.categories):
            idx = i + 1
            if idx <= 10:
                key = str(idx % 10)
                parent = row1
            elif idx <= 20:
                key = f"Sh+{str(idx % 10)}"
                parent = row2
            elif idx <= 30:
                key = f"Ct+{str(idx % 10)}"
                parent = row3
            else:
                break

            lbl = tk.Label(
                parent,
                text=f"[{key}] {cat}",
                bg="#3d3d3d",
                fg="#56b6c2",
                font=("Arial", 10, "bold"),
                padx=5,
                pady=2,
            )
            lbl.pack(side=tk.LEFT, padx=2, pady=2)
            self.category_labels[cat] = lbl

        lbl_fav = tk.Label(
            row1,
            text="[F] ⭐ Favorite",
            bg="#e5c07b",
            fg="black",
            font=("Arial", 10, "bold"),
            padx=5,
            pady=2,
        )
        lbl_fav.pack(side=tk.LEFT, padx=10, pady=2)
        self.category_labels["Favorites"] = lbl_fav

    def setup_bindings(self):
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Control-z>", lambda e: self.undo_last_action())
        self.root.bind("<m>", lambda e: self.mark_selected_reviewed())
        self.root.bind("<M>", lambda e: self.mark_selected_reviewed())
        self.root.bind("<Control-m>", lambda e: self.mark_all_reviewed())
        self.root.bind("<Control-M>", lambda e: self.mark_all_reviewed())
        self.root.bind("<Escape>", lambda e: self.on_escape())
        self.root.bind("<n>", lambda e: self._hotkey_toggle_new_only())
        self.root.bind("<N>", lambda e: self._hotkey_toggle_new_only())
        self.root.bind("<f>", lambda e: self.favorite_image())
        self.root.bind("<F>", lambda e: self.favorite_image())
        self.root.bind("<plus>", lambda e: self._adjust_zoom(0.1))
        self.root.bind("<equal>", lambda e: self._adjust_zoom(0.1))
        self.root.bind("<minus>", lambda e: self._adjust_zoom(-0.1))
        self.root.bind("<KP_Add>", lambda e: self._adjust_zoom(0.1))
        self.root.bind("<KP_Subtract>", lambda e: self._adjust_zoom(-0.1))

        keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
        shift_symbols = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")"]

        for i, k in enumerate(keys):
            self.root.bind(k, lambda e, idx=i + 1: self.sort_image(idx))
            self.root.bind(
                shift_symbols[i], lambda e, idx=i + 11: self.sort_image(idx)
            )
            self.root.bind(
                f"<Control-{k}>", lambda e, idx=i + 21: self.sort_image(idx)
            )

        self.root.bind("<Configure>", self._on_resize)
        self.gallery_canvas.bind(
            "<MouseWheel>", self._on_gallery_mousewheel
        )
        self.gallery_canvas.bind(
            "<Button-4>", lambda e: self._gallery_yview("scroll", -3, "units")
        )
        self.gallery_canvas.bind(
            "<Button-5>", lambda e: self._gallery_yview("scroll", 3, "units")
        )

    def _on_gallery_mousewheel(self, event):
        self._gallery_yview("scroll", int(-1 * (event.delta / 120)), "units")

    def _on_resize(self, event=None):
        if event and event.widget is not self.root:
            return
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(100, self._on_resize_debounced)

    def _on_resize_debounced(self):
        self._resize_job = None
        if self.view_mode == "gallery":
            self._gallery_schedule_sync(full_relayout=True)
        elif self.view_mode == "slideshow":
            self._show_slideshow()

    def _apply_zoom(self, new_zoom):
        self.zoom_level = max(ZOOM_MIN, min(ZOOM_MAX, float(new_zoom)))
        self.zoom_var.set(self.zoom_level)
        self.lbl_zoom.config(text=f"{int(self.zoom_level * 100)}%")
        if self.view_mode == "gallery":
            self._gallery_cached_width = -1
            self._gallery_schedule_sync(full_relayout=True)
        else:
            self._show_slideshow()

    def _on_zoom_change(self, _value):
        self._apply_zoom(self.zoom_var.get())

    def _adjust_zoom(self, delta):
        self._apply_zoom(self.zoom_level + delta)

    @staticmethod
    def _fill_image_to_square(img, cell_size):
        """Resize and center-crop so the image fully fills a square cell."""
        img_w, img_h = img.size
        if img_w < 1 or img_h < 1:
            return img
        scale = max(cell_size / img_w, cell_size / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = max(0, (new_w - cell_size) // 2)
        top = max(0, (new_h - cell_size) // 2)
        return img.crop((left, top, left + cell_size, top + cell_size))

    # ------------------------------------------------------------------ refresh

    def refresh_view(self, scroll_gallery_to_selection=False, gallery_force=False):
        self._clamp_index()
        self._prune_selection()
        self.save_progress()
        self.update_info_label()
        self.update_stats_label()
        if self.view_mode == "slideshow":
            self._show_slideshow()
        else:
            self._render_gallery(
                scroll_to_selection=scroll_gallery_to_selection,
                force=gallery_force,
            )

    def _prune_selection(self):
        self.selected_indices = {
            i for i in self.selected_indices if 0 <= i < len(self.image_files)
        }
        if self.current_index not in self.selected_indices and self.image_files:
            self.selected_indices.add(self.current_index)

    def clear_selection(self):
        if not self.image_files:
            self.selected_indices = set()
            return
        self.selected_indices = {self.current_index}
        if self.view_mode == "gallery":
            self._gallery_refresh_selection_styles()

    def _clamp_index(self):
        if not self.image_files:
            self.current_index = 0
            return
        if self.current_index >= len(self.image_files):
            self.current_index = len(self.image_files) - 1
        if self.current_index < 0:
            self.current_index = 0

    def update_info_label(self):
        if not self.image_files:
            self.lbl_info.config(text="All images sorted!")
            return
        img_path = self.image_files[self.current_index]
        sel_count = len(self.selected_indices)
        base = (
            f"Image {self.current_index + 1} of {len(self.image_files)}: "
            f"{os.path.basename(img_path)}"
        )
        if sel_count > 1:
            base += f"  [{sel_count} selected]"
        self.lbl_info.config(text=base)

    def _get_action_indices(self):
        if len(self.selected_indices) > 1:
            return sorted(self.selected_indices)
        if self.selected_indices:
            return [min(self.selected_indices)]
        return [self.current_index]

    def update_stats_label(self):
        moved_total = sum(self.session_moved.values())
        done = self.session_passed + moved_total
        total = self.session_total_at_load
        remaining = len(self.image_files)
        new_count = len(self.new_paths_at_load)

        parts = [f"Remaining: {remaining}", f"Passed: {self.session_passed}"]
        if moved_total:
            cat_parts = []
            for cat, count in sorted(self.session_moved.items()):
                label = "⭐" if cat == "Favorites" else cat
                if len(label) > 12:
                    label = label[:10] + ".."
                cat_parts.append(f"{label}:{count}")
            parts.append(f"Moved: {moved_total} ({', '.join(cat_parts)})")
        else:
            parts.append("Moved: 0")
        if self.show_new_only:
            parts.append(f"Filter: NEW ({new_count})")
        else:
            parts.append(f"New: {new_count}")
        if total:
            parts.append(f"Done: {done}/{total}")
        self.lbl_stats.config(text=" | ".join(parts))

    # ------------------------------------------------------------------ slideshow

    def _show_slideshow(self):
        if not self.image_files:
            self.canvas.delete("all")
            self.canvas.configure(scrollregion=(0, 0, 0, 0))
            return

        img_path = self.image_files[self.current_index]
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size

            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()

            if canvas_w < 10 or canvas_h < 10:
                self.root.after(50, self._show_slideshow)
                return

            fit_scale = min(canvas_w / img_w, canvas_h / img_h)
            display_scale = fit_scale * self.zoom_level
            new_w = max(1, int(img_w * display_scale))
            new_h = max(1, int(img_h * display_scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")

            scroll_w = max(canvas_w, new_w)
            scroll_h = max(canvas_h, new_h)
            self.canvas.configure(scrollregion=(0, 0, scroll_w, scroll_h))

            x = scroll_w // 2
            y = scroll_h // 2
            self.canvas.create_image(x, y, image=self.photo, anchor=tk.CENTER)

            if new_w > canvas_w or new_h > canvas_h:
                self.slideshow_vsb.pack(side=tk.RIGHT, fill=tk.Y)
                self.slideshow_hsb.pack(side=tk.BOTTOM, fill=tk.X)
                if scroll_w > canvas_w:
                    self.canvas.xview_moveto(
                        (scroll_w - canvas_w) / 2 / (scroll_w - canvas_w)
                    )
                if scroll_h > canvas_h:
                    self.canvas.yview_moveto(
                        (scroll_h - canvas_h) / 2 / (scroll_h - canvas_h)
                    )
            else:
                self.slideshow_vsb.pack_forget()
                self.slideshow_hsb.pack_forget()
                self.canvas.xview_moveto(0)
                self.canvas.yview_moveto(0)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            if self.current_index < len(self.image_files) - 1:
                self._navigate_to_index(self.current_index + 1)

    # ------------------------------------------------------------------ gallery (virtual scroll)

    def _gallery_cancel_loads(self):
        if self._gallery_load_job:
            self.root.after_cancel(self._gallery_load_job)
            self._gallery_load_job = None
        self._gallery_pending = []

    def _gallery_clear_all_items(self):
        self._gallery_cancel_loads()
        for idx in list(self._gallery_items.keys()):
            self._gallery_unload_item(idx)
        self.gallery_canvas.delete("gallery")

    def _gallery_unload_item(self, idx):
        item = self._gallery_items.pop(idx, None)
        if not item:
            return
        for key in ("border_id", "image_id", "placeholder_id"):
            cid = item.get(key)
            if cid:
                self.gallery_canvas.delete(cid)

    def _gallery_compute_layout(self):
        cell_size = max(40, int(BASE_CELL * self.zoom_level))
        viewport_w = max(self.gallery_canvas.winfo_width(), 100)
        outer = cell_size + 2  # 1px border each side
        stride = outer + GALLERY_PAD
        cols = max(1, (viewport_w - GALLERY_PAD) // stride)
        count = len(self.image_files)
        rows = max(1, (count + cols - 1) // cols) if count else 1
        row_height = stride
        total_w = cols * stride + GALLERY_PAD
        total_h = rows * row_height + GALLERY_PAD
        return {
            "cell_size": cell_size,
            "outer": outer,
            "stride": stride,
            "cols": cols,
            "rows": rows,
            "row_height": row_height,
            "total_w": total_w,
            "total_h": total_h,
        }

    def _gallery_cell_origin(self, idx, layout):
        col = idx % layout["cols"]
        row = idx // layout["cols"]
        x = GALLERY_PAD + col * layout["stride"]
        y = GALLERY_PAD + row * layout["row_height"]
        return x, y

    def _gallery_visible_index_range(self, layout):
        if not self.image_files:
            return 0, -1
        top_frac, bottom_frac = self.gallery_canvas.yview()
        total_h = max(layout["total_h"], 1)
        view_h = max(self.gallery_canvas.winfo_height(), 1)
        y0 = max(0, top_frac * total_h - layout["row_height"] * GALLERY_ROW_BUFFER)
        y1 = min(
            total_h,
            bottom_frac * total_h + view_h + layout["row_height"] * GALLERY_ROW_BUFFER,
        )
        first_row = max(0, int(y0 // layout["row_height"]))
        last_row = min(layout["rows"] - 1, int(y1 // layout["row_height"]))
        first_idx = first_row * layout["cols"]
        last_idx = min(len(self.image_files) - 1, (last_row + 1) * layout["cols"] - 1)
        return first_idx, last_idx

    def _gallery_relayout(self, preserve_scroll=True, force=False):
        if not self.image_files:
            self._gallery_clear_all_items()
            self._gallery_layout = None
            self._gallery_queue_count = 0
            self.gallery_canvas.configure(scrollregion=(0, 0, 0, 0))
            return

        viewport_w = self.gallery_canvas.winfo_width()
        if viewport_w < 10:
            self.root.after(50, lambda: self._gallery_relayout(preserve_scroll, force))
            return

        new_layout = self._gallery_compute_layout()
        new_count = len(self.image_files)
        old = self._gallery_layout
        if (
            not force
            and old
            and self._gallery_queue_count == new_count
            and old["cols"] == new_layout["cols"]
            and old["cell_size"] == new_layout["cell_size"]
            and old["total_h"] == new_layout["total_h"]
        ):
            for idx in list(self._gallery_items.keys()):
                self._gallery_update_cell_style(idx)
            self._gallery_sync_visible()
            return

        top_frac = self.gallery_canvas.yview()[0] if preserve_scroll else 0.0
        self._gallery_clear_all_items()
        self._gallery_layout = new_layout
        self._gallery_queue_count = new_count
        self.gallery_canvas.configure(
            scrollregion=(
                0,
                0,
                new_layout["total_w"],
                new_layout["total_h"],
            )
        )
        if preserve_scroll:
            self.gallery_canvas.yview_moveto(top_frac)
        self._gallery_sync_visible()

    def _render_gallery(self, scroll_to_selection=False, force=False):
        self._gallery_relayout(
            preserve_scroll=not scroll_to_selection,
            force=force or scroll_to_selection,
        )
        if scroll_to_selection:
            self.see_selected_in_gallery()

    def _gallery_sync_visible(self):
        if not self.image_files or not self._gallery_layout:
            return

        layout = self._gallery_layout
        first_idx, last_idx = self._gallery_visible_index_range(layout)
        if last_idx < first_idx:
            return

        keep = set(range(first_idx, last_idx + 1))
        for idx in list(self._gallery_items.keys()):
            if idx not in keep:
                self._gallery_unload_item(idx)

        to_load = []
        for idx in range(first_idx, last_idx + 1):
            if idx not in self._gallery_items:
                self._gallery_create_cell_shell(idx, layout)
                to_load.append(idx)
            else:
                self._gallery_update_cell_style(idx)

        if to_load:
            self._gallery_pending.extend(i for i in to_load if i not in self._gallery_pending)
            self._gallery_kick_load_queue()

    def _gallery_cell_style(self, idx):
        if idx == self.current_index:
            return "#98c379", 3
        if idx in self.selected_indices:
            return "#56b6c2", 2
        return "#333333", 1

    def _gallery_refresh_selection_styles(self):
        for idx in list(self._gallery_items.keys()):
            self._gallery_update_cell_style(idx)

    def _gallery_create_cell_shell(self, idx, layout):
        x, y = self._gallery_cell_origin(idx, layout)
        outer = layout["outer"]
        border_color, width = self._gallery_cell_style(idx)

        border_id = self.gallery_canvas.create_rectangle(
            x,
            y,
            x + outer,
            y + outer,
            outline=border_color,
            width=width,
            fill="",
            tags=("gallery", "gallery_cell", f"idx_{idx}"),
        )
        placeholder_id = self.gallery_canvas.create_rectangle(
            x + width,
            y + width,
            x + outer - width,
            y + outer - width,
            outline="",
            fill="#2a2a2a",
            tags=("gallery", "gallery_placeholder", f"idx_{idx}"),
        )
        self._gallery_items[idx] = {
            "border_id": border_id,
            "placeholder_id": placeholder_id,
            "image_id": None,
            "photo": None,
        }

    def _gallery_update_cell_style(self, idx):
        item = self._gallery_items.get(idx)
        layout = self._gallery_layout
        if not item or not layout:
            return
        border_color, width = self._gallery_cell_style(idx)
        self.gallery_canvas.itemconfig(
            item["border_id"], outline=border_color, width=width
        )
        x, y = self._gallery_cell_origin(idx, layout)
        outer = layout["outer"]
        self.gallery_canvas.coords(
            item["border_id"], x, y, x + outer, y + outer
        )
        if item.get("placeholder_id"):
            self.gallery_canvas.coords(
                item["placeholder_id"],
                x + width,
                y + width,
                x + outer - width,
                y + outer - width,
            )
        if item.get("image_id"):
            cx = x + outer // 2
            cy = y + outer // 2
            self.gallery_canvas.coords(item["image_id"], cx, cy)

    def _gallery_kick_load_queue(self):
        if self._gallery_load_job:
            return
        self._gallery_process_load_queue()

    def _gallery_process_load_queue(self):
        self._gallery_load_job = None
        if not self._gallery_pending or not self._gallery_layout:
            return

        batch = self._gallery_pending[:GALLERY_LOAD_PER_TICK]
        self._gallery_pending = self._gallery_pending[GALLERY_LOAD_PER_TICK:]

        for idx in batch:
            if idx in self._gallery_items and self._gallery_items[idx].get("image_id") is None:
                self._gallery_load_thumbnail(idx)

        if self._gallery_pending:
            self._gallery_load_job = self.root.after(1, self._gallery_process_load_queue)

    def _gallery_load_thumbnail(self, idx):
        if idx >= len(self.image_files) or idx not in self._gallery_items:
            return
        layout = self._gallery_layout
        if not layout:
            return

        img_path = self.image_files[idx]
        item = self._gallery_items[idx]
        x, y = self._gallery_cell_origin(idx, layout)
        outer = layout["outer"]
        cell_size = layout["cell_size"]
        _, width = self._gallery_cell_style(idx)

        try:
            with Image.open(img_path) as opened:
                img = opened.convert("RGB")
            inner = max(1, outer - 2 * width)
            img = self._fill_image_to_square(img, inner)
            photo = ImageTk.PhotoImage(img)
            if item.get("placeholder_id"):
                self.gallery_canvas.delete(item["placeholder_id"])
                item["placeholder_id"] = None
            image_id = self.gallery_canvas.create_image(
                x + outer // 2,
                y + outer // 2,
                image=photo,
                anchor=tk.CENTER,
                tags=("gallery", "gallery_cell", f"idx_{idx}"),
            )
            item["image_id"] = image_id
            item["photo"] = photo
            self.gallery_canvas.tag_raise(item["border_id"])
        except Exception as e:
            print(f"Error loading thumbnail {img_path}: {e}")
            if item.get("placeholder_id"):
                self.gallery_canvas.itemconfig(item["placeholder_id"], fill="#3d2020")

    def _on_gallery_canvas_click(self, event):
        if not self.image_files:
            return
        canvas_x = self.gallery_canvas.canvasx(event.x)
        canvas_y = self.gallery_canvas.canvasy(event.y)
        clicked = self.gallery_canvas.find_closest(canvas_x, canvas_y)
        if not clicked:
            return
        for cid in clicked:
            tags = self.gallery_canvas.gettags(cid)
            for tag in tags:
                if tag.startswith("idx_"):
                    index = int(tag[4:])
                    ctrl = bool(event.state & 0x4) or bool(event.state & 0x20000)
                    shift = bool(event.state & 0x1)
                    self.on_gallery_click(
                        index, additive=ctrl and not shift, range_select=shift
                    )
                    return

    def _on_gallery_canvas_double_click(self, event):
        if not self.image_files:
            return
        canvas_x = self.gallery_canvas.canvasx(event.x)
        canvas_y = self.gallery_canvas.canvasy(event.y)
        clicked = self.gallery_canvas.find_closest(canvas_x, canvas_y)
        if not clicked:
            return
        for cid in clicked:
            tags = self.gallery_canvas.gettags(cid)
            for tag in tags:
                if tag.startswith("idx_"):
                    self.enter_slideshow_at(int(tag[4:]))
                    return

    def _on_slideshow_click(self, event=None):
        self.return_to_gallery()

    def _hotkey_toggle_new_only(self):
        self.new_only_var.set(not self.new_only_var.get())
        self.toggle_new_only()

    def _gallery_shift_select_indices(self, index):
        """Shift+click: up-to range on one row; full rows when spanning rows."""
        anchor = self._selection_anchor
        if anchor is None or anchor < 0 or anchor >= len(self.image_files):
            anchor = self.current_index
        layout = self._gallery_layout
        if not layout:
            lo, hi = min(anchor, index), max(anchor, index)
            return set(range(lo, hi + 1))

        cols = layout["cols"]
        anchor_row = anchor // cols
        click_row = index // cols
        if anchor_row == click_row:
            lo, hi = min(anchor, index), max(anchor, index)
            return set(range(lo, hi + 1))

        indices = set()
        for row in range(min(anchor_row, click_row), max(anchor_row, click_row) + 1):
            start = row * cols
            end = min(len(self.image_files), start + cols)
            indices.update(range(start, end))
        return indices

    def on_gallery_click(self, index, additive=False, range_select=False):
        if not self.image_files:
            return
        index = max(0, min(index, len(self.image_files) - 1))

        if range_select:
            new_sel = self._gallery_shift_select_indices(index)
            if additive:
                self.selected_indices |= new_sel
            else:
                self.selected_indices = new_sel
            self.current_index = index
            self.update_info_label()
            self._gallery_refresh_selection_styles()
            self.see_selected_in_gallery()
            return

        if additive:
            if index in self.selected_indices:
                self.selected_indices.discard(index)
                if not self.selected_indices:
                    self.selected_indices.add(index)
            else:
                self.selected_indices.add(index)
            self.current_index = index
            self._selection_anchor = index
            self.update_info_label()
            self._gallery_refresh_selection_styles()
            return

        self.current_index = index
        self.selected_indices = {index}
        self._selection_anchor = index
        self.update_info_label()
        self._gallery_refresh_selection_styles()
        self.see_selected_in_gallery()

    def see_selected_in_gallery(self):
        if not self.image_files or self.view_mode != "gallery" or not self._gallery_layout:
            return
        layout = self._gallery_layout
        row = self.current_index // layout["cols"]
        y = GALLERY_PAD + row * layout["row_height"]
        total_h = layout["total_h"]
        view_h = max(self.gallery_canvas.winfo_height(), 1)
        if total_h <= view_h:
            return
        top_frac = max(0, min(1, (y - layout["row_height"]) / (total_h - view_h)))
        self.gallery_canvas.yview_moveto(top_frac)
        self._gallery_schedule_sync()

    # ------------------------------------------------------------------ navigation / pass

    def _navigate_to_index(self, new_index):
        if not self.image_files:
            return
        new_index = max(0, min(new_index, len(self.image_files) - 1))
        if new_index == self.current_index:
            return
        self.current_index = new_index
        self.selected_indices = {new_index}
        self._selection_anchor = new_index
        self.refresh_view(
            scroll_gallery_to_selection=(self.view_mode == "gallery"),
            gallery_force=False,
        )

    def _pass_images(self, start_idx, end_idx_exclusive):
        """Pass a contiguous slice (used by mark-all)."""
        if start_idx >= end_idx_exclusive or not self.image_files:
            return
        end_idx_exclusive = min(end_idx_exclusive, len(self.image_files))
        indices = list(range(start_idx, end_idx_exclusive))
        self._pass_indices(indices)

    def _pass_indices(self, indices):
        indices = sorted({i for i in indices if 0 <= i < len(self.image_files)})
        if not indices:
            return
        paths = [self.image_files[i] for i in indices]
        for path in paths:
            self.passed_paths.add(path)
            self.new_paths_at_load.discard(path)
        self.undo_stack.append(("pass_indices", paths, indices))
        for i in sorted(indices, reverse=True):
            del self.image_files[i]
        self.session_passed += len(paths)
        self.selected_indices = set()
        self._clamp_index()
        if self.image_files:
            self.selected_indices = {self.current_index}

    def mark_selected_reviewed(self):
        if not self.image_files:
            return
        indices = self._get_action_indices()
        self._pass_indices(indices)
        self.refresh_view(gallery_force=True)

    def mark_all_reviewed(self):
        if not self.image_files:
            return
        count = len(self.image_files)
        if not messagebox.askyesno(
            "Mark all reviewed?",
            f"Mark all {count} remaining images as reviewed (no file moves)?",
        ):
            return
        self._pass_images(0, count)
        self.current_index = 0
        self.selected_indices = set()
        self.refresh_view(gallery_force=True)

    def next_image(self):
        if not self.image_files:
            return
        if self.current_index < len(self.image_files) - 1:
            self._navigate_to_index(self.current_index + 1)

    def prev_image(self):
        if not self.image_files:
            return
        if self.current_index > 0:
            self._navigate_to_index(self.current_index - 1)

    # ------------------------------------------------------------------ folder / load

    def select_input_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder to Sort")
        if not folder:
            return
        if self.current_dir and self.current_dir != folder:
            self.save_progress(commit_seen=True)
        self.current_dir = folder
        self.lbl_input_disp.config(text=f"Input: {self.current_dir}")
        self.load_images()

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Destination Folder")
        if not folder:
            return
        self.gold_standard_dir = folder
        self.lbl_dest_disp.config(text=f"Dest: {self.gold_standard_dir}")
        self.load_categories()
        self.update_legend()

    def _rebuild_queue_from_filters(self, preserve_selection=False):
        prev_path = None
        if (
            preserve_selection
            and self.image_files
            and 0 <= self.current_index < len(self.image_files)
        ):
            prev_path = self.image_files[self.current_index]

        remaining = [
            p for p in self.all_images_at_load if p not in self.passed_paths
        ]
        if self.show_new_only:
            self.image_files = [
                p for p in remaining if p in self.new_paths_at_load
            ]
        else:
            self.image_files = remaining

        self._gallery_cached_width = -1
        self._gallery_queue_count = -1
        self.selected_indices = set()

        if not self.image_files:
            self.current_index = 0
            self.refresh_view(gallery_force=True)
            return

        if prev_path and prev_path in self.image_files:
            self.current_index = self.image_files.index(prev_path)
        else:
            self.current_index = 0
        self.selected_indices = {self.current_index}
        self._selection_anchor = self.current_index
        self.refresh_view(
            scroll_gallery_to_selection=True, gallery_force=True
        )

    def load_images(self):
        all_images = []
        for root_dir, dirs, files in os.walk(self.current_dir):
            for f in files:
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
                ):
                    all_images.append(os.path.join(root_dir, f))

        entry = self._folder_progress_entry() if self.current_dir else {}
        self.passed_paths = set(entry.get("passed", []))
        self.seen_paths = set(entry.get("seen", []))
        self.session_passed = 0
        self.session_moved = defaultdict(int)
        self.session_total_at_load = len(all_images)
        self.undo_stack = []
        self.all_images_at_load = all_images
        self.new_paths_at_load = {
            p
            for p in all_images
            if p not in self.seen_paths and p not in self.passed_paths
        }

        self._gallery_cached_width = -1
        self._gallery_queue_count = -1
        self.selected_indices = set()

        remaining = [p for p in all_images if p not in self.passed_paths]
        if self.show_new_only:
            self.image_files = [
                p for p in remaining if p in self.new_paths_at_load
            ]
        else:
            self.image_files = remaining

        if not self.image_files:
            if all_images and self.show_new_only and remaining:
                self.lbl_info.config(
                    text="No new images since last session. Uncheck New only to see all."
                )
            elif all_images:
                self.lbl_info.config(text="All images sorted!")
            else:
                messagebox.showinfo("Done", "No images found in this directory.")
                self.lbl_info.config(text="Select an input folder to begin...")
            self.refresh_view()
            return

        saved_index = entry.get("index", 0)
        if self.show_new_only:
            self.current_index = 0
        else:
            self.current_index = min(saved_index, len(self.image_files) - 1)
            self.current_index = max(self.current_index, 0)
        self.selected_indices = {self.current_index}
        self._selection_anchor = self.current_index
        self.refresh_view()

        if self.new_paths_at_load and not self.show_new_only:
            n = len(self.new_paths_at_load)
            self.lbl_info.config(
                text=(
                    f"{self.lbl_info.cget('text')}  —  {n} new since last session "
                    f"(press N or use New only)"
                )
            )

    # ------------------------------------------------------------------ sort / undo

    def sort_image(self, key_idx):
        if not self.image_files:
            return
        if key_idx > len(self.categories):
            return
        target_category = self.categories[key_idx - 1]
        self._move_images_to(target_category, self._get_action_indices())

    def favorite_image(self):
        if not self.image_files:
            return
        self._move_images_to("Favorites", self._get_action_indices())

    def _move_images_to(self, target_category, indices):
        if not self.gold_standard_dir:
            messagebox.showerror("Error", "Please Select Destination Folder first!")
            return

        indices = sorted({i for i in indices if 0 <= i < len(self.image_files)})
        if not indices:
            return

        if (
            hasattr(self, "category_labels")
            and target_category in self.category_labels
        ):
            lbl = self.category_labels[target_category]
            original_bg = lbl.cget("bg")
            original_fg = lbl.cget("fg")
            lbl.config(bg="#98c379", fg="black")
            self.root.after(
                400,
                lambda l=lbl, bg=original_bg, fg=original_fg: l.config(
                    bg=bg, fg=fg
                )
                if l.winfo_exists()
                else None,
            )

        target_dir = os.path.join(self.gold_standard_dir, target_category)
        os.makedirs(target_dir, exist_ok=True)

        if target_category not in self.categories:
            self.load_categories()
            self.update_legend()

        moves = []
        try:
            for i in sorted(indices, reverse=True):
                current_img_path = self.image_files[i]
                filename = os.path.basename(current_img_path)
                new_path = os.path.join(target_dir, filename)

                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(new_path):
                    new_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
                    counter += 1

                shutil.move(current_img_path, new_path)
                print(f"Moved -> [{target_category}] {os.path.basename(new_path)}")
                moves.append((new_path, current_img_path, target_category, i))
                self.new_paths_at_load.discard(current_img_path)
                del self.image_files[i]

            moves.reverse()
            if len(moves) == 1:
                new_path, original_path, cat, _orig_idx = moves[0]
                self.undo_stack.append(("move", new_path, original_path, cat))
            else:
                self.undo_stack.append(("move_batch", moves))
            self.session_moved[target_category] += len(moves)
            self.selected_indices = set()
            self._clamp_index()
            if self.image_files:
                self.selected_indices = {self.current_index}
            self.refresh_view(gallery_force=True)

        except Exception as e:
            messagebox.showerror("Move Error", f"Failed to move file: {e}")

    def undo_last_action(self):
        if not self.undo_stack:
            print("Nothing to undo.")
            return

        action = self.undo_stack.pop()

        if action[0] == "move":
            _, new_path, original_path, target_category = action
            try:
                os.makedirs(os.path.dirname(original_path), exist_ok=True)
                shutil.move(new_path, original_path)
                print(f"Undo -> Returned {os.path.basename(original_path)}")
                if self.session_moved[target_category] > 0:
                    self.session_moved[target_category] -= 1
                    if self.session_moved[target_category] <= 0:
                        del self.session_moved[target_category]
                self.image_files.insert(self.current_index, original_path)
                if (
                    original_path in self.all_images_at_load
                    and original_path not in self.seen_paths
                ):
                    self.new_paths_at_load.add(original_path)
                self.selected_indices = {self.current_index}
                self.refresh_view(gallery_force=True)
            except Exception as e:
                messagebox.showerror("Undo Error", f"Failed to undo move: {e}")

        elif action[0] == "move_batch":
            _, moves = action
            try:
                for new_path, original_path, target_category, orig_idx in reversed(
                    moves
                ):
                    os.makedirs(os.path.dirname(original_path), exist_ok=True)
                    shutil.move(new_path, original_path)
                    self.image_files.insert(orig_idx, original_path)
                    if (
                        original_path in self.all_images_at_load
                        and original_path not in self.seen_paths
                    ):
                        self.new_paths_at_load.add(original_path)
                    if self.session_moved[target_category] > 0:
                        self.session_moved[target_category] -= 1
                        if self.session_moved[target_category] <= 0:
                            del self.session_moved[target_category]
                restored = [m[3] for m in moves]
                self.current_index = restored[0]
                self.selected_indices = set(restored)
                print(f"Undo -> Returned {len(moves)} files")
                self.refresh_view(gallery_force=True)
            except Exception as e:
                messagebox.showerror("Undo Error", f"Failed to undo move: {e}")

        elif action[0] == "pass_indices":
            _, paths, indices = action
            for path in paths:
                self.passed_paths.discard(path)
                if path in self.all_images_at_load and path not in self.seen_paths:
                    self.new_paths_at_load.add(path)
            for idx, path in zip(indices, paths):
                self.image_files.insert(idx, path)
            self.session_passed -= len(paths)
            self.current_index = indices[0]
            self.selected_indices = set(indices)
            self.refresh_view(gallery_force=True)

        elif action[0] == "pass_batch":
            _, paths, insert_index = action
            for path in paths:
                self.passed_paths.discard(path)
            self.image_files[insert_index:insert_index] = paths
            self.session_passed -= len(paths)
            self.current_index = insert_index
            self.selected_indices = {insert_index}
            self.refresh_view(gallery_force=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = FastSorterApp(root)
    root.mainloop()
