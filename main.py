# main.py

import tkinter as tk
import torch
from gui.train_gui import TrainGUI
from gui.draw_gui import DrawGUI
import logging

# main.py

class MetricsLogger:
    def __init__(self):
        self.training_time = 0.0
        self.cpu_memory_usage = 0.0
        self.gpu_memory_usage = 0.0
        self.inference_time = 0.0

        # Configure logging to file
        logging.basicConfig(
            filename='metrics.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def log_training_metrics(self, training_time, cpu_mem, gpu_mem, lr, batch_size, epochs):
        self.training_time = training_time
        self.cpu_memory_usage = cpu_mem
        self.gpu_memory_usage = gpu_mem
        log_message = (
            f"Training Metrics - "
            f"Learning Rate: {lr}, Batch Size: {batch_size}, Epochs: {epochs}, "
            f"Training Time: {self.training_time:.2f} sec, "
            f"CPU Memory Used: {self.cpu_memory_usage:.2f} MB, "
            f"GPU Memory Used: {self.gpu_memory_usage:.2f} MB"
        )
        print(f"[Metrics] {log_message}")
        logging.info(log_message)

    def log_inference_time(self, inference_time):
        self.inference_time = inference_time
        log_message = f"Inference Time: {self.inference_time*1000:.2f} ms"
        print(f"[Metrics] {log_message}")
        logging.info(log_message)

class MNISTApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST Trainer & Classifier")

        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # Will be assigned after training or loading

        # Initialize Metrics Logger
        self.metrics = MetricsLogger()

        # Create the training GUI
        self.train_gui = TrainGUI(self, app=self, metrics=self.metrics)

    def open_draw_gui(self):
        """
        Opens the DrawGUI window to classify digits with the trained/loaded model.
        """
        if self.model is None:
            tk.messagebox.showerror("Error", "No trained model found. Train or load a model first!")
            return

        draw_window = DrawGUI(self, metrics=self.metrics)
        draw_window.grab_set()  # Makes the draw window modal-ish

if __name__ == "__main__":
    app = MNISTApp()
    app.mainloop()
