import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# BUG: Missing all random seeds (np.random.seed, torch.manual_seed, etc.)
# An AI auditor should flag this instantly.

def load_data():
    # Load some dummy data
    df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    labels = np.random.randint(0, 2, 100)
    return train_test_split(df.values, labels, test_size=0.2)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

def train():
    X_train, X_test, y_train, y_test = load_data()
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Dummy training loop
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()