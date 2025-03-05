import torch

class DNN(torch.nn.Module):
    def __init__(self, nFeatures=28*28, nClasses=10):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(nFeatures, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 96),
            torch.nn.ReLU(),
            torch.nn.Linear(96, nClasses)
        )

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        self.model.eval()
        return self(x).argmax(dim=1)
    
    def train_loop(self, X, y, epochs, batch_size, optimizer, criterion):
        self.model.train()
        for epoch in range(1, epochs + 1):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size].long()
                optimizer.zero_grad()
                output = self(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch} Loss: {loss.item()}")