# Import Data Class & Library
import CustomDataLoaders
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Select cuda if present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate custom wine data
custom_data_loader = CustomDataLoaders.WineClassificationData()
X, y = custom_data_loader.get_features_targets()
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y[['wine_type']])

X = torch.tensor(X.to_numpy(), dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32)[:, -1].long().reshape(-1, 1).to(device)

print("X Shape: ", X.shape, ", y Shape: ", y.shape, "\n")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train Shape: ", X_train.shape, ", X_test Shape: ", X_test.shape,
      "\ny_train Shape: ", y_train.shape, ", y_test Shape: ", y_test.shape, "\n")
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# ==================== Helper ==================== #
def align_clusters(pred_clusters, true_labels):
    """Map cluster IDs to closest true labels (majority vote)."""
    mapping = {}
    for cluster_id in range(pred_clusters.max().item() + 1):
        mask = (pred_clusters == cluster_id)
        if mask.sum() == 0:
            continue
        true_mode = true_labels[mask].mode()[0].item()
        mapping[cluster_id] = true_mode
    return torch.tensor([mapping[c.item()] for c in pred_clusters], device=device)


def clustering_loss(X, centroids, labels):
    """Compute clustering loss (Sum of Squared Errors)."""
    return torch.sum((X - centroids[labels])**2)


# ==================== KMeans (PyTorch style training loop) ==================== #
k = 2
num_epochs = 100
tol = 1e-4

# Random init centroids
rand_indices = torch.randperm(X_train.size(0))[:k]
centroids = X_train[rand_indices].clone()

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

print("\n\n==================== Model Training (K-Means) ====================")
for epoch in range(num_epochs):
    # Assign clusters
    distances = torch.cdist(X_train, centroids)  # [N_train, k]
    train_labels = torch.argmin(distances, dim=1)

    # Update centroids
    new_centroids = torch.stack(
        [X_train[train_labels == j].mean(dim=0) if (train_labels == j).sum() > 0 else centroids[j]
         for j in range(k)]
    )

    # Compute train loss (SSE)
    train_loss = clustering_loss(X_train, centroids, train_labels)
    train_losses.append(train_loss.item())

    # Predict on test set
    test_distances = torch.cdist(X_test, centroids)
    test_labels = torch.argmin(test_distances, dim=1)
    test_loss = clustering_loss(X_test, centroids, test_labels)
    test_losses.append(test_loss.item())

    # Accuracy (after aligning clusters)
    aligned_train_preds = align_clusters(train_labels, y_train.squeeze())
    aligned_test_preds = align_clusters(test_labels, y_test.squeeze())
    train_acc = (aligned_train_preds == y_train.squeeze()).float().mean().item()
    test_acc = (aligned_test_preds == y_test.squeeze()).float().mean().item()
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    # Update centroids
    shift = torch.norm(new_centroids - centroids, dim=1).sum().item()
    centroids = new_centroids

    # Logging
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss.item():.6f}, '
              f'Test Loss: {test_loss.item():.6f}, '
              f'Train Accuracy: {train_acc:.4f}, '
              f'Test Accuracy: {test_acc:.4f}')

    # Convergence check
    # if shift < tol:
    #     print(f"Converged after {epoch+1} epochs")
    #     break

print(f"\nFinal Train Loss: {train_losses[-1]:.6f}")
print(f"Final Test Loss: {test_losses[-1]:.6f}")
print(f"Final Train Accuracy: {train_accuracies[-1]:.4f}")
print(f"Final Test Accuracy: {test_accuracies[-1]:.4f}")


# ==================== Visualization ==================== #
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss (SSE)")
plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss (SSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss (Sum of Squared Errors)")
plt.title("Training and Test Loss over Epochs (K-Means)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy over Epochs (K-Means)")
plt.legend()
plt.grid(True)
plt.show()
