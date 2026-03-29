import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# BUG: Missing PyTorch deterministic flags
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def train_model():
    # Create dummy dataset
    inputs = torch.randn(1000, 10)
    targets = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(inputs, targets)

    # BUG: shuffle=True with num_workers > 0 requires a worker_init_fn 
    # and a seeded generator to be reproducible!
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4
    )

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    print("Training complete. But is it reproducible?")

if __name__ == "__main__":
    # We set the basic seed, but missed the deeper deterministic flags
    torch.manual_seed(42)
    train_model()