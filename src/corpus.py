import torch
from . import word

class Corpus:
    def __init__(self, corpus: str, word_box: word.WordBox):
        self.corpus = []
        self.word_box = word_box
        for token in word.Tokenizer(corpus).generator():
            self.corpus.append(self.word_box.add(token))

    def into_target(self, context_size: int):
        vocab_size = self.word_box.vocab_size()
        X = []
        Y = []
        for i in range(len(self.corpus)):
            token_id = self.corpus[i]
            context = []
            for k in range(-context_size, context_size + 1):
                if k != 0:
                    context.append(self.corpus[(i + k) % len(self.corpus)])
            X.append(context)
            Y.append([1.0 if k == token_id else 0.0 for k in range(vocab_size)])
        return torch.tensor(X, dtype=torch.long), torch.tensor(Y)

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


