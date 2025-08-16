# Import Data Class & Library
import CustomDataLoaders
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Select cuda if present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate custom wine data
custom_data_loader = CustomDataLoaders.WineRegressionData()
X, y = custom_data_loader.get_features_targets()

X = torch.tensor(X.to_numpy(), dtype=torch.float32)
y = torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1, 1) # Reshape y to (num_samples, 1)
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

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(input_size=X.shape[1], output_size=y.shape[1])
model.to(device)
print("\n==================== Model Summary ====================")
print(summary(model, input_size=X.shape))

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# Lists to store loss plotting
train_losses = []
test_losses = []

print("\n\n==================== Model Training ====================")
# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    # Training phase
    model.train() # Set model to training mode
    
    outputs_train = model(X_train)
    train_loss = criterion(outputs_train, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Evaluation phase (on test set)
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test)
        test_loss = criterion(outputs_test, y_test)
    
    # Store losses
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss.item():.6f}, '
              f'Test Loss: {test_loss.item():.6f}')

print(f"\nFinal Train Loss: {train_losses[-1]:.6f}, Final Test Loss: {test_losses[-1]:.6f}")


print("\n\n==================== Loss Curve ====================")
# Plotting the loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Test Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()