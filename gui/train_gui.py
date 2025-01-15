# gui/train_gui.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import torch
import os

from model import SimpleNN
from train_utils import train_model

class TrainGUI(tk.Frame):
    def __init__(self, parent, app):
        """
        parent: The parent widget (usually the main Tk object).
        app: Reference to the main application (MNISTApp).
        """
        super().__init__(parent)
        self.app = app
        self.pack(fill=tk.BOTH, expand=True)

        # Labels
        tk.Label(self, text="Learning Rate:").grid(row=0, column=0, sticky=tk.E, pady=5, padx=5)
        tk.Label(self, text="Batch Size:").grid(row=1, column=0, sticky=tk.E, pady=5, padx=5)
        tk.Label(self, text="Epochs:").grid(row=2, column=0, sticky=tk.E, pady=5, padx=5)

        # StringVars
        self.lr_var = tk.StringVar(value="0.001")
        self.batch_var = tk.StringVar(value="64")
        self.epochs_var = tk.StringVar(value="5")

        # Entries
        tk.Entry(self, textvariable=self.lr_var).grid(row=0, column=1, pady=5, padx=5)
        tk.Entry(self, textvariable=self.batch_var).grid(row=1, column=1, pady=5, padx=5)
        tk.Entry(self, textvariable=self.epochs_var).grid(row=2, column=1, pady=5, padx=5)

        # Buttons
        self.train_button = tk.Button(self, text="Train Model", command=self.start_training)
        self.train_button.grid(row=3, column=0, pady=5, padx=5)

        self.save_button = tk.Button(self, text="Save Model", command=self.save_model, state=tk.DISABLED)
        self.save_button.grid(row=3, column=1, pady=5, padx=5)

        self.load_button = tk.Button(self, text="Load Model", command=self.load_model)
        self.load_button.grid(row=4, column=0, pady=5, padx=5)

        self.draw_button = tk.Button(self, text="Draw & Classify", command=self.app.open_draw_gui, state=tk.DISABLED)
        self.draw_button.grid(row=4, column=1, pady=5, padx=5)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.grid(row=5, column=0, columnspan=2, pady=10)

    def start_training(self):
        """
        Reads hyperparameters, starts a training thread, updates the GUI accordingly.
        """
        try:
            lr = float(self.lr_var.get())
            batch_size = int(self.batch_var.get())
            epochs = int(self.epochs_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric hyperparameters.")
            return

        # Disable the train button during training
        self.train_button.config(state=tk.DISABLED)

        # Initialize or re-initialize the model
        self.app.model = SimpleNN().to(self.app.device)

        # Reset and show progress bar
        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = epochs

        def train_thread():
            def update_progress(epoch, total):
                self.progress_bar["value"] = epoch

            train_model(
                model=self.app.model,
                device=self.app.device,
                lr=lr,
                batch_size=batch_size,
                epochs=epochs,
                progress_callback=update_progress
            )

            messagebox.showinfo("Info", "Training Complete!")
            self.train_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.draw_button.config(state=tk.NORMAL)

        # Run training in a background thread
        threading.Thread(target=train_thread, daemon=True).start()

    def save_model(self):
        """
        Prompt the user for a file path, then save the trained model.
        """
        if self.app.model is None:
            messagebox.showerror("Error", "No model to save. Train or load a model first.")
            return

        # Prompt for file path
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if file_path:
            torch.save(self.app.model.state_dict(), file_path)
            messagebox.showinfo("Info", f"Model saved to {file_path}")

    def load_model(self):
        """
        Prompt the user for a .pth file, load it into a new model instance.
        """
        file_path = filedialog.askopenfilename(
            title="Select a PyTorch model (.pth)",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if file_path:
            if not os.path.exists(file_path):
                messagebox.showerror("Error", "File does not exist.")
                return

            # Create a new model and load state
            self.app.model = SimpleNN().to(self.app.device)
            self.app.model.load_state_dict(torch.load(file_path, map_location=self.app.device))
            self.app.model.eval()

            messagebox.showinfo("Info", f"Model loaded from {file_path}")
            self.save_button.config(state=tk.NORMAL)
            self.draw_button.config(state=tk.NORMAL)
