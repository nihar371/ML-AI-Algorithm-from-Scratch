# Import Data Class & Library
import CustomDataLoaders
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    correct = (y_pred == y_true).sum().item()
    accuracy = (correct / y_true.size(0)) * 100
    return accuracy


# Select cuda if present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate custom wine data
custom_data_loader = CustomDataLoaders.WineClassificationData()
X, y = custom_data_loader.get_features_targets()
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y[['wine_type']])

X = torch.tensor(X.to_numpy(), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)[:, -1].long().reshape(-1, 1)  # long for classification

print("X Shape: ", X.shape, ", y Shape: ", y.shape, "\n")

# Divide into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train Shape: ", X_train.shape, ", X_test Shape: ", X_test.shape,
      "\ny_train Shape: ", y_train.shape, ", y_test Shape: ", y_test.shape, "\n")

# Move data to device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# ==================== kNN Implementation ==================== #
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data (no actual training like NN)."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict labels for given X."""
        # Expand dimensions for broadcasting: [N_test, 1, D] - [1, N_train, D]
        distances = torch.cdist(X, self.X_train)  # [N_test, N_train]
        
        # Get indices of k smallest distances
        knn_indices = distances.topk(self.k, largest=False).indices  # [N_test, k]

        # Get corresponding labels
        knn_labels = self.y_train[knn_indices].squeeze()  # [N_test, k]

        # Majority vote
        preds = []
        for row in knn_labels:
            values, counts = torch.unique(row, return_counts=True)
            pred = values[counts.argmax()]
            preds.append(pred)
        return torch.stack(preds).reshape(-1, 1)


# ==================== Training & Evaluation ==================== #
k = 5
model = KNNClassifier(k=k)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracies
train_accuracy = calculate_accuracy(y_train, y_train_pred)
test_accuracy = calculate_accuracy(y_test, y_test_pred)

print("\n==================== kNN Results ====================")
print(f"Train Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")


# ==================== Visualization ==================== #
# For kNN there is no loss curve, but we can show accuracy across k values
train_accs, test_accs = [], []
k_values = list(range(1, 16))

for k in k_values:
    model = KNNClassifier(k=k)
    model.fit(X_train, y_train)
    train_accs.append(calculate_accuracy(y_train, model.predict(X_train)))
    test_accs.append(calculate_accuracy(y_test, model.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accs, label="Train Accuracy")
plt.plot(k_values, test_accs, label="Test Accuracy")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy (%)")
plt.title("kNN Accuracy vs k")
plt.legend()
plt.grid(True)
plt.show()