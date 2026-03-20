import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from pathlib import Path

# --- Configuration ---
PROCESSED_DIR = Path("./data/processed")
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
TARGET_COMMANDS = ["forward", "backward", "left", "right", "stop", "_unknown_", "_silence_"]

# --- 1. Define the CNN Architecture ---
class SpeechCommandCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SpeechCommandCNN, self).__init__()
        # Input shape: (Batch, 1, 40, 32)
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (16, 20, 16)
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (32, 10, 8)
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (64, 5, 4)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3), # Helps prevent overfitting
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- 2. Load and Prepare Data ---
def load_data():
    print("Loading processed tensors from disk...")
    X = np.load(PROCESSED_DIR / "X_features.npy")
    y = np.load(PROCESSED_DIR / "y_labels.npy")
    
    # PyTorch expects the channel dimension second: (Batch, Channels, Height, Width)
    # Currently X is (Batch, 40, 32, 1). We need (Batch, 1, 40, 32)
    X = np.transpose(X, (0, 3, 1, 2))
    
    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, y_test

# --- 3. Training Loop ---
def train_model():
    train_loader, test_loader, y_test_true = load_data()
    
    model = SpeechCommandCNN(num_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Training Phase...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad() # Clear old gradients
            
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate error
            
            loss.backward() # Backpropagation
            optimizer.step() # Update weights
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Training Accuracy: {epoch_acc:.2f}%")

    # --- 4. Evaluation ---
    print("\nEvaluating Model on Test Data...")
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            
    # Calculate Metrics
    final_f1 = f1_score(y_test_true, all_preds, average='weighted')
    print("\n" + "="*50)
    print(f"FINAL WEIGHTED F1-SCORE: {final_f1:.4f}")
    print("="*50 + "\n")
    
    print("Detailed Classification Report:")
    print(classification_report(y_test_true, all_preds, target_names=TARGET_COMMANDS))
    
    # Save the trained model
    save_path = MODEL_DIR / "speak_to_steer_cnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved successfully to: {save_path}")

if __name__ == "__main__":
    train_model()