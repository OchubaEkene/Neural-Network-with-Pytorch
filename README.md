# Neural Network with PyTorch

A complete implementation of a Convolutional Neural Network (CNN) for MNIST digit classification using PyTorch. This project includes training, model saving, and image prediction functionality.

## 🚀 Features

- **CNN Architecture**: 3-layer convolutional neural network with ReLU activations
- **MNIST Dataset**: Automatic download and preprocessing of the MNIST dataset
- **Training Pipeline**: Complete training loop with loss tracking
- **Model Persistence**: Save and load trained model weights
- **Image Prediction**: Predict digits from custom images with preprocessing
- **Device Agnostic**: Automatic CPU/GPU detection and usage
- **Confidence Scores**: Detailed prediction confidence and class probabilities

## 📋 Requirements

- Python 3.7+
- PyTorch
- torchvision
- PIL (Pillow)
- NumPy

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/OchubaEkene/Neural-Network-with-Pytorch.git
   cd Neural-Network-with-Pytorch
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision pillow
   ```

## 🏃‍♂️ Usage

### Training the Model

To train the neural network on the MNIST dataset, uncomment the training section in `pytorchnn.py`:

```python
# Uncomment these lines in the main section:
for epoch in range(10): # TRAIN FOR 10 EPOCHS
    for batch in dataset:
        X,y = batch
        X,y = X.to(device), y.to(device)
        yhat = clf(X)
        loss = loss_fn(yhat, y)

        # Backpropagate
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch} loss: {loss.item()}")

with open("model_state.pt", "wb") as f:
    save(clf.state_dict(), f)
```

Then run:
```bash
python pytorchnn.py
```

### Making Predictions

The current script is configured for inference. To predict digits from images:

```bash
python pytorchnn.py
```

This will:
1. Load the trained model weights from `model_state.pt`
2. Process `img_1.jpg` (convert to grayscale, resize to 28x28)
3. Make a prediction and display the results

### Testing with Different Images

To test with other images, simply change the filename in the script:

```python
img = Image.open("img_2.jpg")  # or "img_3.jpg"
```

## 🏗️ Model Architecture

The neural network consists of:

```
Input: 28x28 grayscale image
├── Conv2d(1, 32, kernel_size=3x3) + ReLU
├── Conv2d(32, 64, kernel_size=3x3) + ReLU  
├── Conv2d(64, 64, kernel_size=3x3) + ReLU
├── Flatten
└── Linear(64*22*22, 10) → Output: 10 classes (digits 0-9)
```

## 📊 Training Results

The model achieves high accuracy on MNIST with decreasing loss over epochs:

```
Epoch 0 loss: 0.02016882225871086
Epoch 1 loss: 0.002969881286844611
Epoch 2 loss: 0.0002930955379270017
...
Epoch 9 loss: 5.587911005022761e-07
```

## 📁 Project Structure

```
Neural-Network-with-Pytorch/
├── pytorchnn.py          # Main neural network implementation
├── model_state.pt        # Trained model weights
├── img_1.jpg            # Sample test image
├── img_2.jpg            # Sample test image  
├── img_3.jpg            # Sample test image
├── data/                # MNIST dataset (auto-downloaded)
├── venv/                # Virtual environment
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## 🔧 Key Components

### ImageClassifier Class
- Inherits from `nn.Module`
- Sequential CNN architecture
- Forward pass implementation

### Training Pipeline
- Adam optimizer with learning rate 1e-3
- CrossEntropyLoss for multi-class classification
- Automatic device detection (CPU/GPU)

### Image Preprocessing
- Grayscale conversion
- Resize to 28x28 pixels
- Tensor conversion and normalization
- Batch dimension addition

## 🎯 Example Output

```
Using device: cpu
Predicted digit: 2
Confidence: 1.0000
All class probabilities: [2.30e-21, 1.79e-25, 1.0, 3.26e-17, ...]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Ochuba Ekene**
- GitHub: [@OchubaEkene](https://github.com/OchubaEkene)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- MNIST dataset creators
- The open-source community for inspiration and resources

---

⭐ If you found this project helpful, please give it a star!
