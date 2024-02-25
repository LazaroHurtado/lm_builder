import torch.nn as nn

from .config import FeedForwardConfig

class FeedForward(nn.Module):
    '''
    Q1: Why do we set bias to false?
    Q2: Why are there three linear layers?
    Q3: Why are we doing (act(x@W1) * (x@W))@W2?
    
    Answers to all theses questions can be found in this paper:
    https://arxiv.org/pdf/2002.05202v1.pdf
    '''

    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        
        self.in_dim = config.embedding_dimension
        self.hidden_dim = config.intermediate_dimension
        
        self.w1 = nn.Linear(self.in_dim, self.hidden_dim, bias=False)
        self.activation_fn = config.activation_fn
        self.w2 = nn.Linear(self.hidden_dim, self.in_dim, bias=False)
        self.w3 = nn.Linear(self.in_dim, self.hidden_dim, bias=False)

        self.config = config
    
    def forward(self, x):
        out1 = self.w1(x)
        out2 = self.w2(x)

        x = self.activation_fn(out1)*out2

        out3 = self.w3(x)
        return out3