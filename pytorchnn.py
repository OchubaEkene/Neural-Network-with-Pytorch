# Import dependencies
# To create the model
import torch
from PIL import Image
from torch import nn, save, load
# To optimize the model
from torch.optim import Adam
# To load datasets
from torch.utils.data import DataLoader 

from torchvision import datasets
from torchvision.transforms import ToTensor, Grayscale, Resize, Compose

train_data = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train_data, 32)

# Image Classifier NN
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate the model, loss function, and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
clf = ImageClassifier().to(device)
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Train the model
if __name__ == "__main__":
    # Load the trained model
    with open("model_state.pt", "rb") as f:
        clf.load_state_dict(load(f))
    
    # Set model to evaluation mode
    clf.eval()
    
    # Define image preprocessing pipeline for MNIST-like images
    transform = Compose([
        Grayscale(),  # Convert to grayscale
        Resize((28, 28)),  # Resize to 28x28
        ToTensor()  # Convert to tensor
    ])
    
    # Load and preprocess the image
    img = Image.open("img_1.jpg")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():  # Disable gradient computation for inference
        prediction = clf(img_tensor)
        predicted_class = torch.argmax(prediction, dim=1)
        confidence = torch.softmax(prediction, dim=1)
        
    print(f"Predicted digit: {predicted_class.item()}")
    print(f"Confidence: {confidence[0][predicted_class].item():.4f}")
    print(f"All class probabilities: {confidence[0].tolist()}")
#     for epoch in range(10): # TRAIN FOR 10 EPOCHS
#         for batch in dataset:
#             X,y = batch
#             X,y = X.to(device), y.to(device)
#             yhat =  clf(X)
#             loss = loss_fn(yhat, y)

#             # Backpropagate
#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#         print(f"Epoch {epoch} loss: {loss.item()}")

#     with open("model_state.pt", "wb") as f:
#         save(clf.state_dict(), f)


