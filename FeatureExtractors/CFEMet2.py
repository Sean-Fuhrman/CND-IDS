from torch import nn
from .CNDIDS import ADCNbasic
from .CNDIDS.model import simpleMPL 
from .CNDIDS.memory import Memory
from .CNDIDS.sampler import NNBatchSampler
from .CNDIDS import loss
from .CNDIDS.K_Means import K_Means
import time
import logging
from copy import deepcopy
import torch
from tqdm import tqdm
logger = logging.getLogger()
from pytorch_metric_learning import distances, losses, miners, reducers
from sklearn.cluster import HDBSCAN

class CFEMet2(nn.Module):
    def __init__(self, datastream, config, device):
        super().__init__()
        nNodeInit  = 96
        nOutput   = 500
        nHidden   = 1000
        self.batchSize = config['batch_size']
        self.epoch = config['train_epochs']
        self.model = ADCNbasic.ADCN(datastream.nOutput, nInput=nOutput, nHiddenNode = nNodeInit)
        self.model.ADCNcnn   = simpleMPL(datastream.nInput, nHidden, nOutput)
        self.init_normal = datastream.init_normal
        self.experience_number = 0
        self.layerCount = 0
        
        start_initialization_train = time.time()
        self.model.initialization(self.init_normal, 0, 
                                batchSize = self.batchSize, device = device)

        end_initialization_train = time.time()
        self.labeler = K_Means()

        initialization_time      = end_initialization_train - start_initialization_train
        print("Initialization Time: ", initialization_time)
        

    def fit(self, X, device):
        logger.info("Fitting CFE for experience %d", self.experience_number)
        # store previous task model
        self.model.storeOldModel(self.experience_number)
        sampler = NNBatchSampler(self, X, self.batchSize, nn_per_image=5, is_norm=False)
        distance = distances.LpDistance(p=2, power=1, normalize_embeddings=False)
        reducer = reducers.AvgNonZeroReducer()
        loss_func = losses.TripletMarginLoss(margin=2, distance=distance, reducer=reducer)
        mining_func = miners.TripletMarginMiner(
            margin=2, distance=distance, type_of_triplets="semihard", )
        
        
        cluster_labels = self.labeler.fit_transform(X)
        init_normal_labels = self.labeler.transform(self.init_normal)
        logger.info("Number of clusters: %d", len(set(cluster_labels)))
        logger.info("Number of init_normal clusters: %d", len(set(init_normal_labels)))
    

        y = [1 if i in init_normal_labels else 0 for i in cluster_labels]

        y = torch.tensor(y)
        self.model.loss_func = loss_func
        self.model.mining_func = mining_func
        
        logger.info("----------Beginning CFEM Training for experience %d----------", self.experience_number)
        # train the model
        for j in range(self.epoch):
            for i, batch in enumerate(sampler):
                # drift detection
                x_batch = X[batch]
                label_batch = y[batch]
                self.model.psuedo_labels = label_batch
                if i == 0:
                    prev_batch = None
        
                self.model.driftDetection(x_batch, prev_batch)

                if self.model.driftStatus == 2:
                    logger.info("Drift detected, Adding new layer")
                    
                    # grow layer if drift is confirmed driftStatus == 2
                    self.model.layerGrowing()
                    self.layerCount += 1

                    # initialization phase
                    self.model.initialization(x_batch, self.layerCount, 
                                            batchSize = self.batchSize, device = device)
                    
                if self.model.driftStatus == 0 or self.model.driftStatus == 2:  # only train if it is stable or drift
                    self.model.fit(x_batch)
                    self.model.updateNetProperties()

                    # multi task training
                    if len(self.model.ADCNold) > 0 and self.model.regStrLWF != 0.0:
                        self.model.fitCL(x_batch) 
                prev_batch = x_batch
            logger.info("Finished Epoch %d/%d", j+1, self.epoch)
        logger.info("----------CFEM Training Complete for experience %d----------", self.experience_number)
        self.experience_number += 1
  
    def forward(self, x):
        return self.model.encode(x).cpu()