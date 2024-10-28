import torch
import diversipy
import logging
"""
Memory object

Modes: "Perfect", "FIFO", "PSA"
"""
logger = logging.getLogger()
class Memory(object):
    def __init__(self, mode, capacity, datastream, device):
        self.mode = mode
        self.capacity = capacity 
        self.device = device
        self.memory = torch.tensor([]).to(device)
        if mode == "perfect":
            self.datastream = datastream
    
    def update(self, new_data=None, curr_experience = None, labels=None):
        if self.mode == "perfect" and curr_experience is None:
            raise ValueError("Perfect Memory mode requires current experience")
        
        if self.mode == "perfect":
            logger.info("Updating Perfect Memory")
            self.memory = torch.tensor([]).to(self.device)
            train_experience, _ = self.datastream.train_experiences[curr_experience]
            multiclass_train_labels = self.datastream.multiclass_train_labels[curr_experience]
            attacks = multiclass_train_labels.unique()
            num_examples_per_attack = self.capacity // len(attacks)
            for attack in attacks:
                attack_indices = (multiclass_train_labels == attack).nonzero(as_tuple=True)[0][:num_examples_per_attack]
                attack_data = train_experience[attack_indices].to(self.device)
                self.memory = torch.cat((self.memory, attack_data), dim=0)
        elif self.mode == "FIFO":
            if len(self.memory) + len(new_data) > self.capacity:
                self.memory = torch.cat((self.memory[len(new_data):], new_data), dim=0)
            else:
                self.memory = torch.cat((self.memory, new_data), dim=0)
        elif self.mode == "psa":
            raise NotImplementedError("PSA-Data Memory mode not implemented")
            
    
    def get_memory(self):
        logger.info("Getting Memory")
        return self.memory