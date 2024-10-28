from torch import nn

class PassThroughExtractor(nn.Module):
    def __init__(self, datastream, config):
        super().__init__()

    def fit(self, X, device):
        pass
    
    def forward(self, x):
        return x