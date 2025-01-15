import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ------------------------------------------------
# Step 1: Define or Load Your Model
# ------------------------------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)


# Option 1: If you have a saved model, load its weights here:
model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
model.eval()

# Option 2: If the model is not trained, keep in mind it won't give good results.
#           For demonstration, we'll just skip the actual training part here.

# ------------------------------------------------
# Step 2: Tkinter App to Draw & Classify
# ------------------------------------------------
class DrawDigitApp(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("Draw a Digit and Classify")

        self.model = model
        self.canvas_size = 280  # We'll draw on a 280x280 canvas (10x bigger than MNIST)
        self.brush_size = 10  # Thickness of the stroke

        # Set up Canvas
        self.canvas = tk.Canvas(self, bg="white", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        # Set up Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X)
        classify_btn = tk.Button(btn_frame, text="Classify", command=self.classify_drawing)
        classify_btn.pack(side=tk.LEFT, padx=5, pady=5)

        clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Label to display classification
        self.result_label = tk.Label(self, text="Draw a digit above, then click 'Classify'.")
        self.result_label.pack(pady=5)

        # PIL Image to record the user’s drawing
        self.drawing_image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.drawing_image)

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        """
        Draw on the Tkinter Canvas and the PIL image simultaneously.
        """
        x1 = event.x - self.brush_size
        y1 = event.y - self.brush_size
        x2 = event.x + self.brush_size
        y2 = event.y + self.brush_size

        # Draw on the Tkinter canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")

        # Draw on the PIL image
        self.draw.ellipse([x1, y1, x2, y2], fill=0, outline=0)

    def classify_drawing(self):
        """
        Convert the canvas PIL image to MNIST style, run inference with PyTorch,
        and update the label with the predicted digit.
        """
        # 1) Convert to 28x28 grayscale
        #    (It's already L-mode grayscale, but let's ensure size & invert if needed.)
        img = self.drawing_image.copy()

        # Optional: If your MNIST training was with white text on black background,
        # you might want to invert here. (MNIST digits are white text on black)
        # But let's assume black on white is okay for now.
        # If needed: img = ImageOps.invert(img)

        # Resize from 280x280 down to 28x28
        img = img.resize((28, 28))

        # (Critical for the model classification) Invert: if your MNIST is white-on-black, invert black-on-white drawings
        from PIL import ImageOps
        img = ImageOps.invert(img)

        # 2) Transform to tensor & normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        tensor_img = transform(img).unsqueeze(0).to(device)  # shape [1, 1, 28, 28]

        # 3) Run the model
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor_img)
            # Convert logits to probabilities via softmax
            probs = F.softmax(logits, dim=1)
            # Get predicted class
            pred = probs.argmax(dim=1).item()

        # 4) Display result
        probability_str = ", ".join([f"{i}: {p:.2f}" for i, p in enumerate(probs.cpu().numpy()[0])])
        self.result_label.config(
            text=f"Predicted digit: {pred}\n\nProbabilities:\n{probability_str}"
        )

    def clear_canvas(self):
        """
        Clear the canvas and the PIL drawing image.
        """
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=255)


# ------------------------------------------------
# Step 3: Start the App
# ------------------------------------------------
if __name__ == "__main__":
    app = DrawDigitApp(model)
    app.mainloop()
