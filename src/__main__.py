from . import word
from . import corpus

wordbox = word.WordBox()
corpus = corpus.Corpus("Hello, World! in python")
wordbox.add_from_corpus(corpus)

print(corpus)
print(wordbox)
