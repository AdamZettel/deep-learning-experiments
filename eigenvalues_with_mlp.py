import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Function to generate dataset of symmetric 2x2 matrices with known eigenvalues
def generate_data(num_samples):
    X = np.zeros((num_samples, 2, 2))  # Initialize array for matrices
    y = np.zeros((num_samples, 2))      # Initialize array for eigenvalues
    
    for i in range(num_samples):
        # Generate a random 2x2 symmetric matrix
        A = np.random.randn(2, 2)
        A = (A + A.T) / 2  # Make it symmetric

        # Compute eigenvalues analytically
        eigenvalues, _ = np.linalg.eig(A)
        
        # Store matrix and eigenvalues
        X[i] = A
        y[i] = sorted(eigenvalues)
        
    return X, y

# Generate data and split into train-test sets
np.random.seed(42)
X, y = generate_data(20000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Define a simple MLP model
class EigenvaluePredictor(nn.Module):
    def __init__(self):
        super(EigenvaluePredictor, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4, 10)  # 4 input features (2x2 matrix flattened), 10 hidden units
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 2)  # 2 output units for eigenvalues

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = EigenvaluePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
batch_size = 32
train_losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss / len(X_train))
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]}')

# Evaluation on test set
exact = np.linspace(-3,3,100)
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    test_loss = criterion(outputs, y_test)
    print(f'Test Loss: {test_loss.item()}')

    # Plotting predicted versus actual eigenvalues for all test samples
    actual_eigenvalues = y_test.numpy()
    predicted_eigenvalues = model(X_test).numpy()

    plt.figure(figsize=(10, 8))
    plt.scatter(actual_eigenvalues[:, 0], predicted_eigenvalues[:, 0], marker='o', color='b', label='Eigenvalue 1')
    plt.scatter(actual_eigenvalues[:, 1], predicted_eigenvalues[:, 1], marker='x', color='g', label='Eigenvalue 2')
    plt.plot(exact, exact, color='r', linestyle='-', label='Ideal Eigenvalue 1')
    plt.title('Predicted vs Actual Eigenvalues')
    plt.xlabel('Actual Eigenvalues')
    plt.ylabel('Predicted Eigenvalues')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plotting training loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_losses, color='b', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
