import Stemmer

'''
    Nepali Stemmer Helper Class
    Stemmer is implementation from snowballstem.org
    Credits:  http://www.nepalinlp.com/, http://snowballstem.org/
'''


class NepaliStemmer:
    def __init__(self):
        self.nepali_stemmer = Stemmer.Stemmer('nepali') #initializing nepali stemmer

    # take in a list of tokenized sentences
    # sample input data
    # @input = [[ ['w1'], ['w2'] ], -- one sentence
    #           [ ['v1'], ['v2'] ], 
    #           ... 
    #           [ ['z1'], ['z2'] ]
    #          ]
    # @param t_corpus = tokenized corpus
    def stem_corpus(self, t_corpus):
        s_corpus = []
        for sentence in t_corpus:
            s_corpus.append(self.stem_sentence(sentence))
        return s_corpus

    # single sentence array fed into stemmer
    # [ [w1], [w2], ... , [wn] ] 
    # wn is words
    def stem_sentence(self, t_sentence):
        return self.nepali_stemmer.stemWords(t_sentence)