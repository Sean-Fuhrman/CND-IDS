from torch import nn
from .CNDIDS import ADCNbasic
from .CNDIDS.model import simpleMPL 
from .CNDIDS.memory import Memory
from .CNDIDS.sampler import NNBatchSampler
from .CNDIDS import loss
from .CNDIDS import K_Means
import time
import logging
from copy import deepcopy
import torch
from tqdm import tqdm
logger = logging.getLogger()
from pytorch_metric_learning import distances, losses, miners, reducers
from sklearn.cluster import HDBSCAN

class CFE(nn.Module):
    def __init__(self, datastream, config, device):
        super().__init__()
        nNodeInit  = 96
        nOutput   = 500
        nHidden   = 1000
        self.batchSize = config['batch_size']
        self.epoch = 1
        self.model = ADCNbasic.ADCN(datastream.nOutput, nInput=nOutput, nHiddenNode = nNodeInit)
        self.model.ADCNcnn   = simpleMPL(datastream.nInput, nHidden, nOutput)
        self.init_normal = datastream.init_normal
        self.experience_number = 0
        self.layerCount = 0
        
        start_initialization_train = time.time()
        self.model.initialization(self.init_normal, 0, 
                                batchSize = self.batchSize, device = device)

        end_initialization_train = time.time()
        initialization_time      = end_initialization_train - start_initialization_train
        print("Initialization Time: ", initialization_time)
        

    def fit(self, X, device):
        logger.info("Fitting CFE for experience %d", self.experience_number)
        # store previous task model
        self.model.storeOldModel(self.experience_number)
        # self.train_ADCN(X, device)
        sampler = NNBatchSampler(self, X, self.batchSize, nn_per_image=5, using_feat=True, is_norm=False)
        distance = distances.LpDistance(p=2, power=1, normalize_embeddings=False)
        reducer = reducers.AvgNonZeroReducer()
        loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
        # loss_func = losses.SelfSupervisedLoss(loss_func)
        mining_func = miners.TripletMarginMiner(
            margin=0.2, distance=distance, type_of_triplets="semihard", )
        
        
        labeler = HDBSCAN()
        labels = labeler.fit_predict(X)

        logger.info("Number of clusters: %d", len(set(labels)))
        logger.info("Number of noise points: %d", list(labels).count(-1))
        
        labels = torch.tensor(labels)
        
        self.model.loss_func = loss_func
        self.model.mining_func = mining_func
        
        logger.info("----------Beginning CFEM Training for experience %d----------", self.experience_number)
        # train the model
        for i, batch in enumerate(sampler):
            # drift detection
            x_batch = X[batch]
            label_batch = labels[batch]
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
                self.model.fit(x_batch, epoch=50)
                self.model.updateNetProperties()

                # multi task training
                if len(self.model.ADCNold) > 0 and self.model.regStrLWF != 0.0:
                    self.model.fitCL(x_batch) 
            prev_batch = x_batch
            
        logger.info("----------CFEM Training Complete for experience %d----------", self.experience_number)
        self.experience_number += 1
  
    def forward(self, x):
        return self.model.encode(x).cpu()