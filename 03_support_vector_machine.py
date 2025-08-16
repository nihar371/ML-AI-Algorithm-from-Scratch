# Import Data Class & Library
import CustomDataLoaders
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

def hinge_loss(outputs, labels):
    # Ensure labels are -1 or 1
    labels = labels.float()
    # Hinge loss: max(0, 1 - y * f(x))
    loss = torch.mean(torch.clamp(1 - labels * outputs, min=0))
    return loss

def calculate_accuracy(y_true, y_pred):
    correct = (y_pred.round() == y_true).sum().item()
    accuracy = (correct / y_true.size(0)) * 100
    return accuracy

# Select cuda if present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate custom wine data
custom_data_loader = CustomDataLoaders.WineClassificationData()
X, y = custom_data_loader.get_features_targets()
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y[['wine_type']])
y = np.where(y==0, -1, 1)

X = torch.tensor(X.to_numpy(), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)[:, -1].reshape(-1, 1)
print("X Shape: ", X.shape, ", y Shape: ", y.shape, "\n")

# Divide into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train Shape: ", X_train.shape, ", X_test Shape: ", X_test.shape,
      "\ny_train Shape: ", y_train.shape, ", y_test Shape: ", y_test.shape, "\n")

# Move data to the selected device (CPU or GPU)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Define a simple Logistic Regression model
class SVMModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SVMModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

model = SVMModel(input_size=X.shape[1], output_size=y.shape[1])
model.to(device)
print("\n==================== Model Summary ====================")
print(summary(model, input_size=X.shape))

# Loss and optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

print("\n\n==================== Model Training ====================")
# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    # Training phase
    model.train() # Set model to training mode
    
    outputs_train_logits = model(X_train)
    train_loss = hinge_loss(outputs_train_logits, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Evaluation phase (on test set)
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations for evaluation
        outputs_test_logits = model(X_test)
        test_loss = hinge_loss(outputs_test_logits, y_test)

        # Calculate accuracy
        # Apply sigmoid to logits to get probabilities
        predicted_probs_train = torch.sigmoid(outputs_train_logits)
        predicted_classes_train = (predicted_probs_train >= 0.5).float() # Threshold at 0.5
        train_accuracy = accuracy_score(y_train.cpu().numpy(), predicted_classes_train.cpu().numpy())

        predicted_probs_test = torch.sigmoid(outputs_test_logits)
        predicted_classes_test = (predicted_probs_test >= 0.5).float() # Threshold at 0.5
        test_accuracy = accuracy_score(y_test.cpu().numpy(), predicted_classes_test.cpu().numpy())

    # Store losses and accuracies
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss.item():.6f}, '
              f'Test Loss: {test_loss.item():.6f}, '
              f'Train Accuracy: {train_accuracy:.4f}, '
              f'Test Accuracy: {test_accuracy:.4f}')

print(f"\nFinal Train Loss: {train_losses[-1]:.6f}")
print(f"Final Test Loss: {test_losses[-1]:.6f}")
print(f"Final Train Accuracy: {train_accuracies[-1]:.4f}")
print(f"Final Test Accuracy: {test_accuracies[-1]:.4f}")


print("\n\n==================== Loss & Accuracy Curves ====================")
# Plotting the loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Hinge Loss')
plt.title('Training and Test Loss over Epochs (SVM)')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the accuracy curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy over Epochs (SVM)')
plt.legend()
plt.grid(True)
plt.show()