from Tokenizer import Tokenizer
from Stemmer import Stemmer

tok = Tokenizer()
tokenized_corpus = tok.tokenize_corpus(['वातावरण प्रदुषण न्युनिकरण नगरौ, कुरा हैन काम गरौ" भन्ने नाराका साथ'])

ste = Stemmer()
s = ste.stem_corpus(tokenized_corpus)
# ste.stem_sentence(tokenized_corpus)
print(s)