# main.py

import tkinter as tk
import torch
from gui.train_gui import TrainGUI
from gui.draw_gui import DrawGUI

class MNISTApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST Trainer & Classifier")

        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # Will be assigned after training or loading

        # Create the training GUI
        self.train_gui = TrainGUI(self, app=self)

    def open_draw_gui(self):
        """
        Opens the DrawGUI window to classify digits with the trained/loaded model.
        """
        if self.model is None:
            tk.messagebox.showerror("Error", "No trained model found. Train or load a model first!")
            return

        draw_window = DrawGUI(self)
        draw_window.grab_set()  # Makes the draw window modal-ish

if __name__ == "__main__":
    app = MNISTApp()
    app.mainloop()
