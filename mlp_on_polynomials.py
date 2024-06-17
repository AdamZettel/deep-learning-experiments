import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the polynomial function
def polynomial(x):
    return 3 * x**3 - 2 * x**2 + x - 5

# Generate dataset
np.random.seed(42)
torch.manual_seed(42)

x = np.linspace(-10, 10, 1000).astype(np.float32)
y = polynomial(x)

# Add some noise
noise = np.random.normal(0, 10, y.shape)
y_noisy = y + noise

# Convert to PyTorch tensors
x_tensor = torch.tensor(x,  dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y_noisy,  dtype=torch.float32).view(-1, 1)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = MLP()

# Set up loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1000
train_losses = []

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    train_losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(x_test).numpy()

# Calculate test loss
test_loss = criterion(torch.tensor(y_pred), y_test).item()
print(f'Test Loss: {test_loss:.4f}')

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x_test.numpy(), y_test.numpy(), color='blue', label='Actual')
plt.scatter(x_test.numpy(), y_pred, color='red', alpha=0.5, label='Predicted')
plt.title('MLP Predictions vs Actual')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
