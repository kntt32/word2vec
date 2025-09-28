import torch
from . import word

class Corpus:
    def __init__(self, corpus: str, word_box: word.WordBox):
        self.corpus = []
        self.word_box = word_box
        for token in word.Tokenizer(corpus).generator():
            self.corpus.append(self.word_box.add(token))

    def to_long_tensor(self):
        return torch.tensor([self.corpus], dtype=torch.long)

    def __getitem__(self, index: int):
        return self.word_box[self.corpus[index]]

    def __str__(self):
        string = ""
        first_flag = True
        
        for i in self.corpus:
            token = self.word_box[i]
            token_str = str(token)
            if not first_flag and token_str.isalnum():
                string += " "
            string += token_str

            first_flag = False

        return string


