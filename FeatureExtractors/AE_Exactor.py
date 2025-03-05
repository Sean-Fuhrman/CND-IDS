import torch
import logging
import torch.nn as nn
from copy import deepcopy as deepclone
logger = logging.getLogger()

class AE_Extractor(nn.Module):
    def __init__(self, nFeatures=28*28, nLatent=30, train_epochs=10):
        super().__init__()
        self.train_epochs = train_epochs

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
        self.old_models = []
        self.reg_strength = 1.0
 
    def forward(self, x):
        return self.encoder(x).detach()
    
    def decode(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def LwFloss(self, currentBatchOutput, currentBatchData):
        loss = []
        criterion = nn.MSELoss()
        if len(self.old_models) == 0:
            return 0
        for iTask,_ in enumerate(self.old_models):
            with torch.no_grad():
                minibatch_xOld = self.old_models[iTask].forward(currentBatchData)     # it acts as the target
                
            loss.append(self.reg_strength*criterion(currentBatchOutput, minibatch_xOld))
        sum_loss = sum(loss)
        logger.info(f"LwF Loss: {sum_loss}")
        return sum(loss)
    
    def fit(self, x, device="cuda"):
        epochs = self.train_epochs
        self.train()
        batch_size = 256
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = torch.nn.MSELoss()
        x = x.to(device)
        self.to(device)
        for epoch in range(1, epochs + 1):
            for i in range(0, len(x), batch_size):
                X_batch = x[i:i+batch_size]
                optimizer.zero_grad()
                output = self.decode(X_batch)
                loss = criterion(output, X_batch)
                logger.info(f"AE Loss: {loss.item()}")
                lossLwF = self.LwFloss(self.encoder(X_batch), X_batch)
                loss += lossLwF
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch} Loss: {loss.item()}")
                
        self.old_models.append(deepclone(self))
        self.eval()
        self.cpu()
