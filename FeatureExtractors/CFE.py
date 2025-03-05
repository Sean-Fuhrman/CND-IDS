from torch import nn
from .CNDIDS import ADCNbasic
from .CNDIDS.model import simpleMPL 
from .CNDIDS.memory import Memory
import time
import logging
import torch
logger = logging.getLogger()

class CFE(nn.Module):
    def __init__(self, datastream, config, device):
        super().__init__()
        nNodeInit  = 96
        nOutput   = 500
        nHidden   = 1000
        self.batchSize = config['batch_size']
        self.epoch = 1
        self.model = ADCNbasic.ADCN(datastream.nOutput, nInput=nOutput, nHiddenNode = nNodeInit, desiredLabels=[0,1])
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
        
        if config['memory_mode'] is not None:
            self.memory = Memory(config['memory_mode'], 1000, datastream, device)
        else:
            self.memory = None

    def fit(self, X, device):
        logging.info("Fitting CFE for experience %d", self.experience_number)
        # store previous task model
        self.model.storeOldModel(self.experience_number)
    
        # train the model
        for i in range(0, len(X), self.batchSize):
            x_batch = X[i:i+self.batchSize]
            prev_batch = X[i-self.batchSize:i]
            if self.memory is not None and i != 0:
                x_batch = torch.concatenate((x_batch, self.memory.get_memory().cpu()), axis=0)
            # drift detection
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
                self.model.fit(x_batch, epoch = self.epoch, metricLoss=False)
                self.model.updateNetProperties()

                # multi task training
                if len(self.model.ADCNold) > 0 and self.model.regStrLWF != 0.0:
                    self.model.fitCL(x_batch, reconsLoss=True) 
                    
            if self.memory is not None:
                self.memory.update(new_data=x_batch, curr_experience=self.experience_number)        
        self.experience_number += 1
                
    def forward(self, x):
        return self.model.encode(x).cpu()