# MNIST Digit Classifier with GUI

A Python application for training, saving, loading, and using a neural network to classify handwritten digits from the MNIST dataset. The application features a Tkinter-based GUI for easy interaction.

## Features

- Train neural networks with customizable hyperparameters
- Save and load trained models
- Interactive drawing canvas for digit classification
- Real-time probability distribution display
- User-friendly GUI interface

## Project Structure
```
my_mnist_app/
│
├── main.py                # Application entry point
├── model.py              # PyTorch model definitions
├── train_utils.py        # Training utilities
├── gui/
│   ├── train_gui.py      # Training interface
│   └── draw_gui.py       # Drawing and classification interface
│
├── README.md
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/my_mnist_app.git
cd my_mnist_app
```

2. Create and activate virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install torch torchvision pillow
# Or use requirements.txt
pip install -r requirements.txt
```

## Usage

### Running the Application
```bash
python main.py
```

### Training a Model
1. Set hyperparameters:
   - Learning rate
   - Batch size
   - Number of epochs
2. Click "Train Model"
3. Monitor training progress
4. Save model when complete

### Using the Drawing Interface
1. Click "Draw & Classify"
2. Draw a digit in the canvas
3. Click "Classify" to get predictions
4. Use "Clear" to reset the canvas

## Troubleshooting

### SSL Certificate Errors
If encountering SSL errors during MNIST download:
```bash
pip install --upgrade certifi
```

### Poor Classification Results
- Ensure digits are centered in the canvas
- Try increasing training epochs
- Adjust learning rate and batch size
- Verify image preprocessing

## Tips for Best Results
- Train for multiple epochs to improve accuracy
- Use appropriate learning rates (e.g., 0.001)
- Draw digits clearly and centered
- Save models regularly to avoid retraining

## License

MIT License

## Acknowledgements
- PyTorch
- Tkinter
- MNIST Dataset

---
For more information or to report issues, please visit the project repository.
