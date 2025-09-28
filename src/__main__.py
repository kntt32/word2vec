import torch
from torch import nn
from . import word
from . import corpus
from . import model
from . import trainer

wordbox = word.WordBox()
corpus = corpus.Corpus("Hello, World! Hello in python", wordbox)

print(corpus)
print(corpus.corpus)

model = model.Word2Vec(3)
model.init_params(wordbox.vocab_size())

X = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 0], [0, 4, 5]])
Y = torch.tensor([
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float)
trainer = trainer.Trainer(model)
trainer.train(X, Y, 2000)

print(torch.softmax(model.forward(X), dim=1))


