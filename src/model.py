import torch
from torch import nn
from . import word
from . import corpus

class Word2Vec(nn.Module):
    def __init__(self, embedding_dim: int):
        nn.Module.__init__(self)
        self.embedding_dim = embedding_dim

    def init_params(self, vocab_size: int):
        self.in_embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.out_embedding = nn.Linear(self.embedding_dim, vocab_size, bias = False)

    def forward(self, x): # [batch_size, context_len]
        embedded = self.in_embedding(x) # [batch_size, context_len, embedding_dim]
        context_vector = torch.mean(embedded, dim=1) # [batch_size, embedding_dim]
        y = self.out_embedding(context_vector) # [batch_size, vocab_size]
        return y
        
