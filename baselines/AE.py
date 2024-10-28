import torch
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger()

class AE(torch.nn.Module):
    def __init__(self, nFeatures=28*28, nLatent=9):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(nFeatures, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 96),
            torch.nn.ReLU(),
            torch.nn.Linear(96, nLatent)
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nLatent, 96),
            torch.nn.ReLU(),
            torch.nn.Linear(96, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, nFeatures),
            torch.nn.Sigmoid()
        )
        self.mean = None
        self.std = None
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def predict(self, x):
        x = x.view(x.size(0), -1)
        encoded = self(x)
        return ((encoded - x)**2).mean(dim=1)
    
    def fit(self, x, epochs=10, device="cuda"):
        x = x.to(device)
        self.train()
        batch_size = 256
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        criterion = torch.nn.MSELoss()
        for epoch in range(1, epochs + 1):
            for i in range(0, len(x), batch_size):
                X_batch = x[i:i+batch_size].to(device)
                optimizer.zero_grad()
                output = self(X_batch)
                loss = criterion(output, X_batch)
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch} Loss: {loss.item()}")
        self.eval()
