import torch
import torch.nn as nn
import logging
from pytorch_metric_learning import distances, losses, miners, reducers
from tqdm import tqdm
from sklearn.cluster import HDBSCAN
from .CNDIDS.K_Means import K_Means
from copy import deepcopy as deepclone

logger = logging.getLogger()

class Met2LwF(torch.nn.Module):
    def __init__(self, datastream, nLatent=128):
        super().__init__()
        self.datastream = datastream
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.datastream.nInput, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, nLatent)
        )

        self.reg_strength = 0.1
        
        self.distance = distances.LpDistance(p=2, power=1)
        self.reducer = reducers.ThresholdReducer(low = 0)
        self.loss_func = losses.TripletMarginLoss(margin=2, distance=self.distance, reducer=self.reducer)
        self.mining_func = miners.TripletMarginMiner(margin=2, distance=self.distance)
        self.labeler = K_Means()
        
        self.old_models = []
        self.reg_strength = 1.0
 
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
            currentBatchOutput = currentBatchOutput/torch.norm(currentBatchOutput, dim=1).reshape(-1,1)
            minibatch_xOld = minibatch_xOld/torch.norm(minibatch_xOld, dim=1).reshape(-1,1)
            currentBatchData = currentBatchData.to(self.device)
            minibatch_xOld = minibatch_xOld.to(self.device)
            loss.append(self.reg_strength*criterion(currentBatchOutput, minibatch_xOld))
        sum_loss = sum(loss)
        # logger.info(f"LwF Loss: {sum_loss}")
        return sum_loss

    ## Fit function that combines reconstruction loss with metric loss
    def fit(self, x, device="cuda", epochs=40):
        self.device = device
        self.train()
        batch_size = 128
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        logger.info("Fitting Labeler")
        cluster_labels = self.labeler.fit_transform(x)
        init_normal = self.datastream.init_normal
        
        # Get the cluster labels for the init_normal data
        init_normal_labels = self.labeler.transform(init_normal)
        logger.info("Number of clusters: %d", len(set(cluster_labels)))
        logger.info("Number of init_normal clusters: %d", len(set(init_normal_labels)))
                
        # map cluster labels to one if they are in the init_normal_labels
        y = [1 if i in init_normal_labels else 0 for i in cluster_labels]
        
        x = x.float().to(device)
        y = torch.tensor(y).to(device)
        
        x_val = x[:len(x)//10]
        y_val = y[:len(y)//10]
        
        x = x[len(x)//10:]
        y = y[len(y)//10:]
        
        self.to(device)
        logger.info("-------- Starting EaM Train Loop--------")
        
        best_loss = float('inf')
        best_epoch = 0
        save_path = "best_model.pth"
        
        for epoch in range(1, epochs + 1):
            losses = []
            normal_losses = []
            Lwf_losses = []
            for i in tqdm(range(0, len(x), batch_size)):
                X_batch = x[i:i+batch_size]
                Y_batch = y[i:i+batch_size]
                optimizer.zero_grad()
                embeddings = self.encoder(X_batch.reshape(X_batch.size()[0],-1))
                indices_tuple = self.mining_func(embeddings, Y_batch)
                loss = self.loss_func(embeddings, Y_batch, indices_tuple)
                normal_losses.append(loss.item())
                lossLwF = self.LwFloss(embeddings, X_batch)
                Lwf_losses.append(lossLwF)
                loss += lossLwF
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            logger.info("Epoch %d, Loss: %f, Normal Loss: %f, Lwf Loss: %f", epoch, sum(losses)/len(losses), sum(normal_losses)/len(normal_losses), sum(Lwf_losses)/len(Lwf_losses))
            
            x_val = x_val.float().to(device)
            y_val = y_val.to(device)
            self.eval()
            with torch.no_grad():
                losses = []
                for i in range(0, len(x_val), batch_size):
                    X_batch = x_val[i:i+batch_size]
                    Y_batch = y_val[i:i+batch_size]
                    embeddings = self.encoder(X_batch.reshape(X_batch.size()[0],-1))
                    indices_tuple = self.mining_func(embeddings, Y_batch)
                    loss = self.loss_func(embeddings, Y_batch, indices_tuple)
                    lossLwF = self.LwFloss(embeddings, X_batch)
                    loss += lossLwF
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
        