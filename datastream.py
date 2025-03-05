import torch
import numpy as np
import utils
from copy import deepcopy
import logging

logger = logging.getLogger()

class datastream:
    def __init__(self, train_experiences, test_experiences, init_normal, nInput, name, normalize=False):
        self.train_experiences = train_experiences
        self.test_experiences = test_experiences

        self.multiclass_train_labels = [deepcopy(experience[1]) for experience in train_experiences]
        self.multiclass_test_labels = [deepcopy(experience[1]) for experience in test_experiences]

        self.init_normal = init_normal[:int(len(init_normal) * 0.9)]
        self.init_val = init_normal[int(len(init_normal) * 0.9):]
        self.nInput = nInput
        self.nOutput = 2
        self.nExperiences = len(train_experiences)
        self.name = name
        self.convert_experiences_to_binary()
        if normalize:
            self.normalize_data()

    def get_KMeans_subset(self, experience_num, nEachClassSamples = 500):
        one_mask = self.train_experiences[0][1] == 1
        zero_mask = self.train_experiences[0][1] == 0

        # Not random so that it is the same subset for each task
        first_attacks = np.unique(self.multiclass_train_labels[0][one_mask])
        selected_attack = first_attacks[0]
        selected_attack_mask = self.multiclass_train_labels[0] == selected_attack
        one_indices = selected_attack_mask.nonzero().squeeze()[:nEachClassSamples]
        zero_indices = zero_mask.nonzero().squeeze()[:nEachClassSamples]
        labeled_indices = torch.cat((one_indices, zero_indices)).long()
        
        selected_attacks, counts = np.unique(self.multiclass_train_labels[0][labeled_indices], return_counts=True)
        logger.info("Task %d: Selected Attacks: %s, Counts: %s", experience_num, selected_attacks, counts)

        taskLabeledData    = self.train_experiences[0][0][labeled_indices]
        taskLabeledLabel   = self.train_experiences[0][1][labeled_indices]
        return taskLabeledData, taskLabeledLabel

    def get_ADCN_datastream(self, label_mode = "random"):   
        return dataADCNLoader(self.train_experiences, self.test_experiences, self.multiclass_train_labels, label_mode, self.nInput, batchSize = 1000, nEachClassSamples = 500)

    def convert_experiences_to_binary(self):
        normal_attack = utils.get_normal_attack(self.name)
        for i, experience in enumerate(self.train_experiences):
            X, Y = experience
            mask = torch.isin(Y, normal_attack)
            Y[mask] = 0
            Y[~mask] = 1
            self.train_experiences[i] = (X, Y)
        for i, experience in enumerate(self.test_experiences):
            X, Y = experience
            mask = torch.isin(Y, normal_attack)
            Y[mask] = 0
            Y[~mask] = 1
            self.test_experiences[i] = (X, Y)

    def normalize_data(self):
        # Normalize data so that every column is [0,1)
        max = torch.max(self.init_normal, dim=0)[0]
        min = torch.min(self.init_normal, dim=0)[0]
        diff = max - min
        diff = torch.where(diff == 0, 1, diff)
        self.init_normal = (self.init_normal - min) / (diff)
        self.init_val = (self.init_val - min) / (diff)

        logger.info("Normalizing datastream %s", self.name)
        
        for i, experience in enumerate(self.train_experiences):
            X, Y = experience
            X = (X - min) / (diff)
            X = torch.clamp(X, 0, 1)
            self.train_experiences[i] = (X, Y)
        for i, experience in enumerate(self.test_experiences):
            X, Y = experience
            X = (X - min) / (diff)
            X = torch.clamp(X, 0, 1)
            self.test_experiences[i] = (X, Y)

class dataADCNLoader(object):
    def __init__(self, train_experiences, test_experiences, multiclass_labels, label_mode,  nInput, batchSize = 1000, nEachClassSamples = 500):
        self.train_experiences  = train_experiences
        self.test_experiences   = test_experiences
        self.batchSize = batchSize
        self.nEachClassSamples = nEachClassSamples
        self.nOutput = 2
        self.label_mode = label_mode
        self.nInput = nInput
        self.multiclass__labels = multiclass_labels
        self.createTask(nEachClassSamples)

    def createTask(self, nEachClassSamples):
        self.nTask = len(self.train_experiences)
        self.nBatch = 0

        self.taskIndicator        = []

        # init final labeled data
        finalLabeledData          = {}
        finalLabeledLabel         = {}

        # init final unlabeled data
        finalUnlabeledData        = {}
        finalUnlabeledLabel       = {}

        # testing data
        unlabeledDataTest        = {}
        unlabeledLabelTest       = {}

        nUnlabeledDataTest = 0
        logger.info("creating ADCN datastream with mode: %s", self.label_mode)
        for iTask in range(0,self.nTask):
            # load data
            # iTask = iTask + 1
            nBatchPerCurrentTask = len(self.train_experiences[iTask][0])//self.batchSize
            self.nBatch += nBatchPerCurrentTask - 1
            self.taskIndicator = self.taskIndicator + (iTask*torch.ones(nBatchPerCurrentTask - 1).long()).tolist()

            # load labeled data
            # get nEachClassSamples samples for each class [0 or 1]
            if self.label_mode == "random":
                one_mask = self.train_experiences[iTask][1] == 1
                zero_mask = self.train_experiences[iTask][1] == 0
                one_indices = torch.where(one_mask)[0]
                zero_indices = torch.where(zero_mask)[0]
                one_indices = one_indices[torch.randperm(one_indices.size(0))[:nEachClassSamples]]
                zero_indices = zero_indices[torch.randperm(zero_indices.size(0))[:nEachClassSamples]]
                labeled_indices = torch.cat((one_indices, zero_indices)).long()

                selected_attacks, counts = np.unique(self.multiclass__labels[iTask][labeled_indices], return_counts=True)
                logger.info("Task %d: Selected Attacks: %s, Counts: %s", iTask, selected_attacks, counts)

                if counts[0] < nEachClassSamples:
                    logger.warning("Not enough samples for attack %d, only %d samples", selected_attacks[0], counts[0])
                    logger.info("selecting half, then duplicating samples for attack %d", selected_attacks[0])
                    one_indices = one_indices[:len(one_indices)//2]
                    while len(one_indices) < nEachClassSamples:
                        one_indices = torch.cat((one_indices, one_indices[:nEachClassSamples-len(one_indices)])).long()
                    labeled_indices = torch.cat((one_indices, zero_indices)).long()
                    selected_attacks, counts = np.unique(self.multiclass__labels[iTask][labeled_indices], return_counts=True)
                    logger.info("Task %d: Selected Attacks: %s, Counts: %s", iTask, selected_attacks, counts)

                taskLabeledData    = self.train_experiences[iTask][0][labeled_indices]
                taskLabeledLabel   = self.train_experiences[iTask][1][labeled_indices]
            elif self.label_mode == "first-experience-only-single-attack":
                one_mask = self.train_experiences[0][1] == 1
                zero_mask = self.train_experiences[0][1] == 0

                # Not random so that it is the same subset for each task
                first_attacks = np.unique(self.multiclass__labels[0][one_mask])
                selected_attack = first_attacks[0]
                selected_attack_mask = self.multiclass__labels[0] == selected_attack
                one_indices = selected_attack_mask.nonzero().squeeze()[:nEachClassSamples]
                zero_indices = zero_mask.nonzero().squeeze()[:nEachClassSamples]
                labeled_indices = torch.cat((one_indices, zero_indices)).long()
                
                selected_attacks, counts = np.unique(self.multiclass__labels[0][labeled_indices], return_counts=True)
                logger.info("Task %d: Selected Attacks: %s, Counts: %s", iTask, selected_attacks, counts)

                taskLabeledData    = self.train_experiences[0][0][labeled_indices]
                taskLabeledLabel   = self.train_experiences[0][1][labeled_indices]

                if iTask != 0:
                    labeled_indices = torch.tensor([])
            elif self.label_mode == "first-experience-only-multi-attack":
                one_mask = self.train_experiences[0][1] == 1
                zero_mask = self.train_experiences[0][1] == 0

                # Not random so that it is the same subset for each task
                first_attacks = torch.unique(self.multiclass__labels[0][one_mask])
                selected_attack_mask = torch.isin(self.multiclass__labels[0], first_attacks)
                one_indices = selected_attack_mask.nonzero().squeeze()[:nEachClassSamples]
                zero_indices = zero_mask.nonzero().squeeze()[:nEachClassSamples]
                labeled_indices = torch.cat((one_indices, zero_indices)).long()
                
                selected_attacks, counts = np.unique(self.multiclass__labels[0][labeled_indices], return_counts=True)
                logger.info("Task %d: Selected Attacks: %s, Counts: %s", iTask, selected_attacks, counts)

                taskLabeledData    = self.train_experiences[0][0][labeled_indices]
                taskLabeledLabel   = self.train_experiences[0][1][labeled_indices]

                if iTask != 0:
                    labeled_indices = torch.tensor([])
            elif self.label_mode == "first-experience-only-multi-attack-no-imbalance":
                one_mask = self.train_experiences[0][1] == 1
                zero_mask = self.train_experiences[0][1] == 0

                # Not random so that it is the same subset for each task
                first_attacks = torch.unique(self.multiclass__labels[0][one_mask])
                nPerClass = nEachClassSamples // len(first_attacks)
                for attack in first_attacks:
                    selected_attack_mask = self.multiclass__labels[0] == attack
                    one_indices = selected_attack_mask.nonzero().squeeze()[:nPerClass]
                    if len(one_indices) < nPerClass:
                        logger.warning("Not enough samples for attack %d, only %d samples", attack, len(one_indices))
                        logger.info("Selecting half, then duplicate samples for attack %d", attack)
                        one_indices = one_indices[:len(one_indices)//2]
                        while len(one_indices) < nPerClass:
                            one_indices = torch.cat((one_indices, one_indices[:nPerClass-len(one_indices)])).long()

                    if attack == first_attacks[0]:
                        labeled_indices = one_indices
                    else:
                        labeled_indices = torch.cat((labeled_indices, one_indices)).long()

                zero_indices = zero_mask.nonzero().squeeze()[:nEachClassSamples]
                labeled_indices = torch.cat((labeled_indices, zero_indices)).long()
                
                selected_attacks, counts = np.unique(self.multiclass__labels[0][labeled_indices], return_counts=True)
                logger.info("Task %d: Selected Attacks: %s, Counts: %s", iTask, selected_attacks, counts)

                taskLabeledData    = self.train_experiences[0][0][labeled_indices]
                taskLabeledLabel   = self.train_experiences[0][1][labeled_indices]
                if iTask != 0:
                    labeled_indices = torch.tensor([])
            else:
                raise ValueError(f"{self.label_mode} is a invalid label_mode")
            
            labeled_indices = list(set(labeled_indices.tolist()))

            # load unlabeled data
            taskUnlabeledData  = utils.deleteRowTensor(self.train_experiences[iTask][0], labeled_indices, mode=2)
            taskUnlabeledLabel = utils.deleteRowTensor(self.train_experiences[iTask][1], labeled_indices, mode=2)

            # store labeled data and labels
            finalLabeledData[iTask]  = taskLabeledData
            finalLabeledLabel[iTask] = taskLabeledLabel

            # store unlabeled data and labels
            finalUnlabeledData[iTask]  = taskUnlabeledData[self.batchSize:]
            finalUnlabeledLabel[iTask] = taskUnlabeledLabel[self.batchSize:]

            # store unlabeled data for testing
            unlabeledDataTest[iTask]  = self.test_experiences[iTask][0]
            unlabeledLabelTest[iTask] = self.test_experiences[iTask][1]  

            nUnlabeledDataTest += unlabeledDataTest[iTask].shape[0]        

        # labeled data
        self.labeledData    = finalLabeledData
        self.labeledLabel   = finalLabeledLabel

        # unlabeled data
        self.unlabeledData  = finalUnlabeledData
        self.unlabeledLabel = finalUnlabeledLabel

        # testing data
        self.unlabeledDataTest  = unlabeledDataTest
        self.unlabeledLabelTest = unlabeledLabelTest