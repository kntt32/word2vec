class Token:
    def __init__(self, word: str):
        self.word = word.strip()

    def as_str(self):
        return self.word

    def __str__(self):
        return self.word

    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return self.word == other.word

    def __hash__(self):
        return hash(self.word)

class Tokenizer:
    def __init__(self, text: str):
        self.text = text

    def generator(self):
        word = ""
        
        for char in self.text:
            if char.isalnum():
                word += char
            else:
                if len(word) != 0:
                    yield Token(word)
                word = ""
                if not char.isspace():
                    yield Token(char)

        if len(word) != 0:
            yield Token(word)

class WordBox:
    def __init__(self):
        self.box = []

    def vocab_size(self):
        return len(self.box)

    def add(self, token: Token):
        if token not in self.box:
            self.box.append(token)
            return len(self.box) - 1
        else:
            return self.box.index(token)

    def add_from_str(self, s: str):
        for token in Tokenizer(s).generator():
            self.add(token)

    def __str__(self):
        string = "WordBox["
        for token in self.box:
            string += "\"" + str(token) + "\"" + ", "
        return string

    def __contains__(self, token: Token):
        return token in self.box

    def __getitem__(self, index):
        return self.box[index]


