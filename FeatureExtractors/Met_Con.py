import torch
import logging
from pytorch_metric_learning import distances, losses, miners, reducers
from tqdm import tqdm
from sklearn.cluster import HDBSCAN
from .modules.sampler import NNBatchSampler
import os
logger = logging.getLogger()

class Met_Con(torch.nn.Module):
    def __init__(self, nFeatures=28*28, batch_size=256, nLatent=30):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(nFeatures, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 96),
            torch.nn.ReLU(),
            torch.nn.Linear(96, nLatent)
        )
         
        self.mean = None
        self.std = None
        
        self.distance = distances.LpDistance(p=2, power=1)
        self.reducer = reducers.ThresholdReducer(low = 0)
        self.loss_func = losses.TripletMarginLoss(margin=2, distance=self.distance, reducer=self.reducer)
        self.loss_func = losses.SelfSupervisedLoss(self.loss_func)
        self.model_path = "Met_Con.pth"
        self.batch_size = batch_size
 
    def forward(self, x):
        return self.encoder(x).detach()
    

    ## Fit function that combines reconstruction loss with metric loss
    def fit(self, x, device="cuda", epochs=20):
        self.train()
   
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        x = x.float().to(device)
        self.to(device)
        x_val = x[:int(0.1*len(x))]
        x = x[int(0.1*len(x)):]
        sampler = NNBatchSampler(self, x, self.batch_size, nn_per_image=5, is_norm=False)
        
        logger.info("-------- Starting EaM Train Loop--------")
        best_loss = 100000
        best_epoch = 0
        for epoch in range(1, epochs + 1):
            losses = []
           
            for i, batch_idx in tqdm(enumerate(sampler)):
                x_batch = x[batch_idx.to(device)]
                optimizer.zero_grad()
                embeddings = self.encoder(x_batch.reshape(x_batch.size()[0],-1))
                loss = self.loss_func(embeddings, embeddings)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            enc_val = self.encoder(x_val.reshape(x_val.size()[0],-1))
            loss_val = self.loss_func(enc_val, enc_val).item()
            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(self.state_dict(), self.model_path)
                best_epoch = epoch
                    
            logger.info("Epoch %d, Loss: %f", epoch, sum(losses)/len(losses))
            
        logger.info("-------- Finished EaM Train Loop--------")
        self.load_state_dict(torch.load(self.model_path))
        logger.info("Best Loss: %f, Best Epoch: %d", best_loss, best_epoch)
        
        #delete the model file
        os.remove(self.model_path)
        self.eval()
        self.cpu()