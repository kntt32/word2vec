import torch
import matplotlib.pyplot as plt
from torch import nn
from . import word
from . import corpus
from . import model

wordbox = word.WordBox()
corpus = corpus.Corpus("Hello, World! Hello in python", wordbox)

print(corpus)
print(corpus.corpus)
print(corpus.into_target(1))

model = model.Word2Vec(3)
model.init_params(wordbox.vocab_size())

X, Y = corpus.into_target(1)
plot_x, plot_y = model.train(X, Y, 200, lr = 1.0, visualize = True)

print(torch.softmax(model.forward(X), dim=1))

plt.plot(plot_x, plot_y)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("loss.png")

