import numpy as np
from gensim.models import Word2Vec as w2v
from nlp_processor import NepaliStemmer,Tokenizer

nep2vec = w2v.load('nep2vec_snowball_stemmer.model')
nepali_stemmer = NepaliStemmer.NepaliStemmer()
nepali_tokenizer = Tokenizer()

TAGSET_URL = "tagset.txt"
with open(TAGSET_URL, 'r') as t:
    tagsets = t.read().split('\n')

# zeros are special
ts_sc_num2tags = { i + 1  : tags for i, tags in enumerate(tagsets) }
ts_sc_tags2num = { tags: i + 1 for i, tags in enumerate(tagsets) }

def tags_encode(t_in, source, seq_length):
    op_data = np.zeros(seq_length)
    for i, tags in enumerate(t_in):
        if i < seq_length:
            try:
                op_data[i] = source[tags]
            except KeyError:
                continue
    return op_data

def tags_decode(t_in, source):
    dec_data = []
    for tags in t_in:
        if (tags != 0.0):
            dec_data.append(source[tags])
    return dec_data

def words_encode(words,seq_len=100,feature_len=100):
    stemmed_words = [words]
    stemmed_words = nepali_stemmer.stem_corpus([words])
    X = np.zeros([seq_len, feature_len])
    for k, token in enumerate(stemmed_words[0]):
        if(token in nep2vec.wv.vocab):
            index = k
            if index >= seq_len:
                continue
            X[index] = nep2vec.wv[token]
    X = X.reshape(1, seq_len, feature_len)
    return X

# print(words_encode(['नेपली']))
