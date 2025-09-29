import torch
from torch import nn
from . import word
from . import corpus

class Word2Vec(nn.Module):
    def __init__(self, embedding_dim: int):
        nn.Module.__init__(self)
        self.embedding_dim = embedding_dim

    def init_params(self, vocab_size: int):
        self.in_embedding = nn.EmbeddingBag(vocab_size, self.embedding_dim, mode="mean")
        self.out_embedding = nn.Linear(self.embedding_dim, vocab_size, bias = False)
        self.layer = nn.Sequential(
            self.in_embedding,
            self.out_embedding,
        )

    def forward(self, x): # [batch_size, context_len]
        y = self.layer(x)
        # embedded = self.in_embedding(x) # [batch_size, context_len, embedding_dim]
        # context_vector = torch.mean(embedded, dim=1) # [batch_size, embedding_dim]
        # y = self.out_embedding(context_vector) # [batch_size, vocab_size]
        return y

    def train(self, inputs: torch.LongTensor, targets: torch.Tensor, epoches, lr = 0.1, visualize = False):
        plot_x = []
        plot_y = []

        loss_fn = nn.CrossEntropyLoss();
        optimizer = torch.optim.SGD(params = self.parameters(), lr = lr)

        for epoch in range(epoches):
            optimizer.zero_grad()
            y = self.forward(inputs) # [batch_size, vocab_size]
            loss = loss_fn(y, targets)
            loss.backward()
            optimizer.step()

            if visualize and epoch % 10 == 0:
                plot_x.append(epoch)
                plot_y.append(loss.item())

        if visualize:
            return plot_x, plot_y





