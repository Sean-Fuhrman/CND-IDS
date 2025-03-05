import torch
import logging
from pytorch_metric_learning import distances, losses, miners, reducers
from tqdm import tqdm
logger = logging.getLogger()

class EaM(torch.nn.Module):
    def __init__(self, nFeatures=28*28, nLatent=30):
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
        return self.encoder(x).detach()
    
    def decode(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

    ## Fit function that combines reconstruction loss with metric loss
    def fit(self, x_normal, x_anomaly, device="cuda", epochs=1):
        self.train()
        metric_lr = 0.5
        recon_lr = 0.5
        distance = distances.LpDistance(p=2, power=1, normalize_embeddings=True)
        reducer = reducers.AvgNonZeroReducer()
        loss_func = losses.AngularLoss(alpha=40, reducer=reducer)
        mining_func = miners.TripletMarginMiner(
            margin=0.2, distance=distance, type_of_triplets="semihard", )
        batch_size = 64
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        if(x_normal.shape[0] > x_anomaly.shape[0]):
            x_normal = x_normal[:x_anomaly.shape[0]]
        else:
            x_anomaly = x_anomaly[:x_normal.shape[0]]
            
            
        for epoch in range(1, epochs + 1):
            for i in tqdm(range(0, len(x_normal), batch_size)):
                x_nb = x_normal[i:i+batch_size]
                x_ab = x_anomaly[i:i+batch_size]
                x_b = torch.cat([x_nb, x_ab], 0)
                optimizer.zero_grad()
                recon_loss = self.get_reconstruction_loss(x_b, criterion)
                mining_loss = self.get_mining_loss(loss_func, mining_func, x_nb, x_ab, optimizer)
                loss = metric_lr * mining_loss + recon_lr * recon_loss
         
                loss.backward()
                optimizer.step()
         
        self.eval()
        self.cpu()

    def get_reconstruction_loss(self, X_batch, criterion):
        output = self.decode(X_batch)
        loss = criterion(output, X_batch)
        return loss

    ### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
    def get_mining_loss(self, loss_func, mining_func, x_nb, x_ab, optimizer):
        real = x_nb
        fake = x_ab
        data = torch.cat([real, fake], 0)
        labels = torch.cat([torch.ones(real.size()[0]), torch.zeros(fake.size()[0])], 0)
        
        shuffle_idx = torch.randperm(data.size()[0])
        data = data[shuffle_idx]
        labels = labels[shuffle_idx]
        embeddings = self(data.reshape(data.size()[0],-1)).cpu()
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        return loss