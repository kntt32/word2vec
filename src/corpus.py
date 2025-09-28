from . import word

class Corpus:
    def __init__(self, corpus: str):
        self.corpus = []
        for token in word.Tokenizer(corpus).generator():
            self.corpus.append(token)

    def __contains__(self, token):
        return token in self.corpus

    def __getitem__(self, index):
        return self.corpus[index]

    def __str__(self):
        string = ""
        first_flag = True
        
        for token in self.corpus:
            token_str = str(token)
            if not first_flag and token_str.isalnum():
                string += " "
            string += token_str

            first_flag = False

        return string


