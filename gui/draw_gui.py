# gui/draw_gui.py

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms
import time

class DrawGUI(tk.Toplevel):
    def __init__(self, app, metrics):
        super().__init__()
        self.app = app
        self.metrics = metrics
        self.title("Draw a Digit and Classify")

        self.canvas_size = 280  # 10x bigger than 28×28
        self.brush_size = 10

        # Canvas
        self.canvas = tk.Canvas(self, bg="white", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        # PIL image
        self.drawing_image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.drawing_image)

        # Mouse binding
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons
        button_frame = tk.Frame(self)
        button_frame.pack(fill=tk.X)

        classify_btn = tk.Button(button_frame, text="Classify", command=self.classify_drawing)
        classify_btn.pack(side=tk.LEFT, padx=5, pady=5)

        clear_btn = tk.Button(button_frame, text="Clear", command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.result_label = tk.Label(self, text="Draw a digit, then click 'Classify'.")
        self.result_label.pack(pady=5)

    def paint(self, event):
        x1 = event.x - self.brush_size
        y1 = event.y - self.brush_size
        x2 = event.x + self.brush_size
        y2 = event.y + self.brush_size

        # Draw on the Tkinter canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")

        # Draw on the PIL image
        self.draw.ellipse([x1, y1, x2, y2], fill=0, outline=0)

    def classify_drawing(self):
        if self.app.model is None:
            messagebox.showerror("Error", "No trained model. Train or load a model first!")
            return

        # Copy
        img = self.drawing_image.copy()
        # Resize
        img = img.resize((28, 28))
        # Invert if ne
