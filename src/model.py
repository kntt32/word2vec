import torch.nn
from . import word
from . import corpus

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.corpus_list = []
        self.word_box = word.WordBox()

    def add(self, corpus):
        self.corpus.append(corpus)
        self.word_box.add_from_corpus(corpus)
