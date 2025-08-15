# Import Data Class & Library
import CustomDataLoaders
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

# Select cuda if present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate custom wine data
custom_data_loader = CustomDataLoaders.WineRegressionData()
X, y = custom_data_loader.get_features_targets()
X = torch.tensor(X.to_numpy(), dtype=torch.float32)
y = torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1, 1)
print("X Shape: ", X.shape, "\nY Shape: ", y.shape, "\n")

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(input_size=X.shape[1], output_size=y.shape[1])
model.to(device)
print("\n=== Model Summary ===")
print(summary(model, input_size=X.shape))

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

print("\n=== Model Training ===")
# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()

    X = X.to(device)
    y = y.to(device)

    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

print(f"\nFinal Loss: {loss.item():.6f}")

print("\n=== Final Weights & Bias ===")
# Print learned parameters
params = list(model.parameters())
print(f"Learned weight:\n{params[0]}\nLearned bias:\n{params[1]}")