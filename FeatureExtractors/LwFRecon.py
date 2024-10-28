import torch
import torch.nn as nn
import logging
from pytorch_metric_learning import distances, losses, miners, reducers
from tqdm import tqdm
from sklearn.cluster import HDBSCAN
from .CNDIDS.K_Means import K_Means
from .CNDIDS.sampler import NNBatchSampler
from copy import deepcopy as deepclone

logger = logging.getLogger()

class Met2LwF(torch.nn.Module):
    def __init__(self, datastream, nLatent=30, train_epochs=20):
        super().__init__()
        self.datastream = datastream
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.datastream.nInput, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, nLatent)
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nLatent, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.datastream.nInput)
        )
        self.distance = distances.LpDistance()
        self.reducer = reducers.ThresholdReducer(low = 0)
        self.loss_func = losses.TripletMarginLoss(margin=2, distance=self.distance, reducer=self.reducer)
        self.mining_func = miners.TripletMarginMiner(margin=2, distance=self.distance, type_of_triplets="semihard")
        
        self.old_models = []
        self.reg_strength = 1
        
        self.LwF_strength = 1
        
        self.train_epochs = train_epochs
 
    def forward(self, x):
        x = x.to(self.device)
        self.encoder.to(self.device)
        return self.encoder(x).detach().cpu()
    
    def LwFloss(self, currentBatchOutput, currentBatchData):
        loss = []
        criterion = nn.MSELoss()
        for iTask,_ in enumerate(self.old_models):
            with torch.no_grad():
                minibatch_xOld = self.old_models[iTask](currentBatchData)     # it acts as the target
            
            #normalize the output
            # currentBatchOutput = currentBatchOutput/torch.norm(currentBatchOutput, dim=1).reshape(-1,1)
            # minibatch_xOld = minibatch_xOld/torch.norm(minibatch_xOld, dim=1).reshape(-1,1)
            currentBatchData = currentBatchData.to(self.device)
            minibatch_xOld = minibatch_xOld.to(self.device)
            loss.append(self.reg_strength*criterion(currentBatchOutput, minibatch_xOld))
        sum_loss = sum(loss)
        # logger.info(f"LwF Loss: {sum_loss}")
        return sum_loss

    def reconstruction_loss(self, embeddings, x):
        self.decoder.to(self.device)
        x = x.to(self.device)
        return torch.nn.functional.mse_loss(self.decoder(embeddings), x)
    
    def metric_loss(self, embeddings, y):
        indices_tuple = self.mining_func(embeddings, y)
        return self.loss_func(embeddings, y, indices_tuple)
    
    def balance_data(self, x, y):
        logger.info("Balancing Data")
        
        from imblearn.over_sampling import SMOTE
        from collections import Counter
        
        smote = SMOTE()
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        
        x, y = smote.fit_resample(x, y)
        
        logger.info("Balanced Data")
        
        x = torch.tensor(x).to(self.device)
        y = torch.tensor(y).to(self.device)
        
        return x, y
    
    ## Fit function that combines reconstruction loss with metric loss
    def fit(self, x, device="cuda"):
        self.device = device
        self.train()
        batch_size = 64
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        x = x.float().to(device)
        x_val = x[:len(x)//5] 
        x = x[len(x)//5:]
    
        self.to(device)
        logger.info("-------- Starting LwFRecon Train Loop--------")
        
        best_loss = float('inf')
        best_epoch = 0
        save_path = "best_model.pth"
        
        for epoch in range(1, self.train_epochs + 1):
            shuffle = torch.randperm(len(x))
            x = x[shuffle]
            losses = []

            Lwf_losses = []
            reconstruction_losses = []
            for i in tqdm(range(0, len(x), batch_size)):
                X_batch = x[i:i+batch_size]
                optimizer.zero_grad()
                embeddings = self.encoder(X_batch.reshape(X_batch.size()[0],-1))
                lossLwF = self.LwFloss(embeddings, X_batch) * self.LwF_strength
                Lwf_losses.append(lossLwF)
                reconstruction_loss = self.reconstruction_loss(embeddings, X_batch) * self.reg_strength
                reconstruction_losses.append(reconstruction_loss.item())
                loss = lossLwF + reconstruction_loss 
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            logger.info("Epoch %d, Loss: %f, Lwf Loss: %f, Recon Loss: %f", epoch, sum(losses)/len(losses), sum(Lwf_losses)/len(Lwf_losses), sum(reconstruction_losses)/len(reconstruction_losses))
            
            x_val = x_val.float().to(device)
            self.eval()
            with torch.no_grad():
                for i in range(0, len(x_val), batch_size):
                    X_batch = x_val[i:i+batch_size]
                    embeddings = self.encoder(X_batch.reshape(X_batch.size()[0],-1))
                    reconstruction_loss = self.reconstruction_loss(embeddings, X_batch) * self.reg_strength
                    loss = lossLwF + reconstruction_loss
                    losses.append(loss.item())
                loss = sum(losses)/len(losses)
                logger.info("Validation Loss: %f", loss)
                if loss < best_loss:
                    best_loss = loss
                    torch.save(self.state_dict(), save_path)
                    logger.info("Saving Model with Loss: %f", best_loss)
                    best_epoch = epoch
        
        self.load_state_dict(torch.load(save_path))
        logger.info("Loading Best Model from Epoch: %d", best_epoch)
        self.eval()
        self.old_models.append(deepclone(self))
        