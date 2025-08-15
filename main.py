import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# =====================
# 1. Load & Prepare Data
# =====================
df = pd.read_csv("riceClassification.csv")  # Replace with your file

# Separate features & labels
X = df.drop(columns=["Class"]).values
y = df["Class"].values

# Train / Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to tensors (float64)
X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
y_train_tensor = torch.tensor(y_train, dtype=torch.float64)
X_val_tensor = torch.tensor(X_val, dtype=torch.float64)
y_val_tensor = torch.tensor(y_val, dtype=torch.float64)

# Create DataLoaders
train_data = TensorDataset(X_train_tensor, y_train_tensor)
val_data = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# =====================
# 2. Define Model
# =====================
class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Output probabilities for binary classification
        )

    def forward(self, x):
        return self.net(x)

# =====================
# 3. Training Setup
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(input_dim=X_train.shape[1]).double().to(device)  # float64 on device
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =====================
# 4. Training Loop
# =====================
epochs = 20
for epoch in range(epochs):
    # Training phase
    model.train()
    total_loss_train, total_correct_train = 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        preds = model(inputs).view(-1)  # Flatten output
        loss = criterion(preds, labels)
        
        total_loss_train += loss.item()
        total_correct_train += (preds.round() == labels).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    total_loss_val, total_correct_val = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            preds = model(inputs).view(-1)
            loss = criterion(preds, labels)
            
            total_loss_val += loss.item()
            total_correct_val += (preds.round() == labels).sum().item()

    # Print epoch results
    print(f"Epoch {epoch+1}: "
          f"Train Loss={total_loss_train/len(train_loader):.4f}, "
          f"Train Acc={total_correct_train/len(X_train):.4f}, "
          f"Val Loss={total_loss_val/len(val_loader):.4f}, "
          f"Val Acc={total_correct_val/len(X_val):.4f}")
