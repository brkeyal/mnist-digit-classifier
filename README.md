# MNIST Digit Classifier with GUI

This Python application allows users to **train**, **save**, **load**, and **use** a neural network to classify handwritten digits from the MNIST dataset. It features a user-friendly **Graphical User Interface (GUI)** built with **Tkinter**, enabling users to:

1. **Input Hyperparameters**: Specify learning rate, batch size, and number of epochs.
2. **Train the Model**: Initiate training based on the provided hyperparameters.
3. **Save/Load Models**: Save trained models to disk and load existing models for inference.
4. **Draw & Classify**: Draw a digit in a canvas and classify it using the trained model, displaying both the predicted digit and the probability distribution across all classes.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Using the Application](#using-the-application)
  - [Training the Model](#training-the-model)
  - [Saving and Loading Models](#saving-and-loading-models)
  - [Drawing and Classifying Digits](#drawing-and-classifying-digits)
- [Notes & Tips](#notes--tips)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Project Structure
my_mnist_app/ │ ├── main.py # Entry point of the application ├── model.py # PyTorch model definitions ├── train_utils.py # Training utilities (data loading, training loop, etc.) ├── gui/ │ ├── train_gui.py # Tkinter GUI for training, saving, loading models │ └── draw_gui.py # Tkinter GUI for drawing and classifying digits │ ├── README.md # This file └── requirements.txt # Python dependencies (optional)



---

## Installation

### Prerequisites

- **Python 3.7+**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/yourusername/my_mnist_app.git
cd my_mnist_app


