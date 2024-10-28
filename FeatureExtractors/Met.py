import torch
import logging
from pytorch_metric_learning import distances, losses, miners, reducers
from tqdm import tqdm
logger = logging.getLogger()

class Met(torch.nn.Module):
    def __init__(self, nFeatures=28*28, nLatent=30):
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
        self.mining_func = miners.TripletMarginMiner(margin=2, distance=self.distance)
 
    def forward(self, x):
        return self.encoder(x)
    

    ## Fit function that combines reconstruction loss with metric loss
    def fit(self, x_normal, x_attack, device="cuda", epochs=5):
        self.train()
        batch_size = 64
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        x = torch.cat([x_normal, x_attack], 0)
        y = torch.cat([torch.zeros(x_normal.size()[0]), torch.ones(x_attack.size()[0])], 0)
        x = x.float().to(device)
        y = y.to(device)
        self.to(device)
        logger.info("-------- Starting EaM Train Loop--------")
        for epoch in range(1, epochs + 1):
            losses = []
            for i in tqdm(range(0, len(x), batch_size)):
                X_batch = x[i:i+batch_size]
                Y_batch = y[i:i+batch_size]
                optimizer.zero_grad()
                embeddings = self(X_batch.reshape(X_batch.size()[0],-1))
                indices_tuple = self.mining_func(embeddings, Y_batch)
                loss = self.loss_func(embeddings, Y_batch, indices_tuple)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            logger.info("Epoch %d, Loss: %f", epoch, sum(losses)/len(losses))
        self.eval()
        self.cpu()