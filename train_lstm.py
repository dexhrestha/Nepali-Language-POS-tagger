import sys
import argparse
from utils_lstm import csv_to_corpus,map_tag2index,convert_tag2index,data_generator
from keras.utils import to_categorical

import gensim.models.word2vec as w2v
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Dropout, Bidirectional, TimeDistributed, Embedding, Activation,Masking
from keras.optimizers import Adam

import pickle
import  numpy as np
# import pandas as pd

argp = argparse.ArgumentParser()

argp.add_argument("--corpus","-c",help="path to corpus file")
argp.add_argument("--labels","-l",help="path to lable file (pcikle)")
argp.add_argument("--tokenizer","-t",help="path to tokenizer file (pcikle)")
argp.add_argument("--w2v","-w",help="w2vfile")

args = argp.parse_args()


# python train_lstm.py --corpus "../Dataset/dataset_cs.csv" --labels "withMaskingtag2index.pickle" --tokenizer "withMaskingTokenizer.pickle" --w2v "../nepw2vmodel"


class POSLSTM:
    def __init__(self):
        self.filename = args.__getattribute__('corpus')
        self.nepw2v = w2v.Word2Vec.load(args.__getattribute__('w2v'))
        self.tokenizer_file = args.__getattribute__('tokenizer')
        self.tokenizer = Tokenizer(lower=False, oov_token='-OOV-')
        self.embedding_matrix = []
        self.model = Sequential()
        self.tag2index= {}
        self.sentences = []
        self.sentence_tags = []
        self.labels = []
        self.vocab_size = 0
        self.max_length = 0

    def create_tokenizer(self):
        self.tokenizer.fit_on_texts(self.sentences)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        encoded_docs = self.tokenizer.texts_to_sequences(self.sentences)
        self.max_length = len(max(self.sentences,key=len))
        # max_length = 197
        self.sentences = pad_sequences(encoded_docs, maxlen=self.max_length, padding='post')
        self.sentence_tags = pad_sequences(self.sentence_tags,maxlen=self.max_length, padding='post')
        self.tokenizer.word_index['-PAD-'] = 0

        with open(self.tokenizer_file,'wb') as f:
            pickle.dump(self.tokenizer,f)

    def create_embedding_matrix(self):
        self.embedding_matrix = np.zeros((self.vocab_size, self.nepw2v.wv.vector_size))
        for word, i in self.tokenizer.word_index.items():
            try:
                embedding_vector = nepw2v.wv.get_vector(word)
            except:
                embedding_vector = None
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

    def process_corpus(self):
        corpus = csv_to_corpus(self.filename)
        for sentence in corpus:
            x=[]
            y=[]
            for word in sentence:
                x.append(word[1])
                y.append(word[0])
            if len(x) > 0 and len(x)<200:
                self.sentences.append(x)
                self.sentence_tags.append(y)
        self.tag2index = map_tag2index(self.sentence_tags,args.__getattribute__('labels'))
        self.sentence_tags = convert_tag2index(self.tag2index,self.sentence_tags)
        self.create_tokenizer()
        self.create_embedding_matrix()
    
    def build_model(self):
            
        self.model.add(InputLayer(input_shape=(self.max_length,)))
        self.model.add(Embedding(self.vocab_size, self.nepw2v.wv.vector_size,weights=[self.embedding_matrix],input_length=self.max_length,trainable=False))
        self.model.add(Masking())
        self.model.add(Bidirectional(LSTM(10, return_sequences=True)))
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(Dense(len(self.tag2index))))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
 
        print(self.model.summary())

    def train_test(self):        
        train_sentences,test_sentences,train_sentence_tags,test_sentence_tags = train_test_split(self.sentences,self.sentence_tags,test_size=0.2)
        # train_generator = data_generator(train_sentences,train_sentence_tags,3000)
        # self.model.fit_generator(train_generator,samples_per_epoch=3000,epochs=10)
        print("Sentnecs :",len(train_sentences))
        print("sentence_tags",to_categorical(train_sentence_tags).shape)
        self.model.fit(train_sentences,to_categorical(train_sentence_tags),epochs=10,validation_split=0.2,batch_size=256)
        self.model.evaluate(test_sentences,to_categorical(test_sentence_tags))


if __name__ == "__main__":
    lstm  = POSLSTM()
    lstm.process_corpus()
    lstm.build_model()
    # print(len(lstm.sentences))
    # lstm.test()
    lstm.train_test()
    
    