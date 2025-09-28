import torch
from torch import nn
from torch import optim

class Trainer:
    def __init__(self, module: nn.Module):
        self.module = module
        self.loss_fn = nn.CrossEntropyLoss();
        self.optimizer = optim.SGD(params = self.module.parameters(), lr = 0.1)

    def train(self, inputs: torch.LongTensor, targets: torch.Tensor, epoches: int, lr = 0.1):
        # inputs: [batch_size, context_len]
        # targets: [batch_size, vocab_size]
        for epoch in range(epoches):
            self.optimizer.zero_grad()
            y = self.module.forward(inputs) # [batch_size, vocab_size]
            loss = self.loss_fn(y, targets)
            loss.backward()
            self.optimizer.step()



