import sys
import argparse

import numpy as np
import pandas as pd
from math import ceil
import random

from pathlib import Path

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Input, Dropout, TimeDistributed, Activation, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from utils_gen import *

argp = argparse.ArgumentParser()

argp.add_argument("--datapath","-c",help="path to corpus files")
argp.add_argument("--labels","-l",help="path to lable file (pcikle)")
argp.add_argument("--tokenizer","-t",help="path to tokenizer file (pcikle)")
argp.add_argument("--w2v","-w",help="w2vfile")

args = argp.parse_args()

class POS():
    def __init__(self,seq_len=100,feature_len=100):
        self.datapath = args.__getattribute__('datapath')
        self.data_list = [] #file names
        self.seq_len = seq_len
        self.feature_len = feature_len
        self.tagsets = tagsets
        self.train_data_list = []
        self.val_data_list = []
        self.test_data_list = []
        self.train_data_length = 0
        self.val_data_length = 0
        self.test_data_length = 0
        self.model = []
        

    def get_filenames(self):
        print("Searching path for files...")
        data_list = list(Path(self.datapath).rglob("*.[cC][sS][vV]"))
        print("Found {0} csv files in {1}".format(len(data_list), self.datapath))
        return data_list
    
    def get_total_data_length(self,d_list=None):
        tot = 0
        for path in d_list:
            df = pd.read_csv(path)
            df = df[["tags","words"]]
            for t, w in zip(df["tags"], df["words"]):
                sentences = w.split("#") 
                if(len(sentences) > self.seq_len):
                    continue
                
                tot = tot + 1
        return tot
    
    def train_test_split(self,d_list,train_split=0.6):
        random.shuffle(d_list)

        x = int(len(d_list) * train_split)
        v = int((len(d_list)-x) * 0.5)

        print(x,v)

        train_data_list = d_list[:x]
        val_data_list = d_list[x:len(d_list)-v]
        test_data_list = d_list[len(d_list)-v:]

        train_data_length = self.get_total_data_length(train_data_list)
        val_data_length = self.get_total_data_length( val_data_list)
        test_data_length = self.get_total_data_length(test_data_list)
        
        print("Train Data Sentences #: {0}".format(train_data_length))
        print("Train Data Sentences #: {0}".format(val_data_length))
        print("Test Data Sentences #: {0}".format(test_data_length))
    
        return train_data_length,val_data_length,test_data_length,train_data_list,val_data_list,test_data_list

        
    def make_generator(self,d_list, print_data=False):
        for path in d_list:
            df = pd.read_csv(path)
            df = df[["tags","words"]]
            for t, w in zip(df["tags"], df["words"]):
                tags = t.split("#")
                sentences = w.split("#")
                
                if(len(sentences) > self.seq_len):
                    continue
                    
                if print_data is True:
                    print(sentences, tags)
                e_sentences = words_encode(sentences).reshape(1,self.seq_len, self.feature_len)
                e_tags = tags_encode(tags, ts_sc_tags2num, self.seq_len)
                e_tags = to_categorical(e_tags, num_classes=len(self.tagsets)+1).reshape(1, self.seq_len, len(self.tagsets)+1)
                
                
                yield e_sentences, e_tags

    def batch_generator(self,generator, batch_size, total_pos):
        i = 0
        batch_st = []
        batch_tag = []
        for e_sentences, e_tags in generator:
            batch_st.append(e_sentences.reshape(self.seq_len, self.feature_len))
            batch_tag.append(e_tags.reshape(self.seq_len, total_pos))
            if i % batch_size == batch_size - 1:
                yield np.asarray(batch_st), np.asarray(batch_tag)
                batch_st = []
                batch_tag = []
            i = i + 1
        yield np.asarray(batch_st), np.asarray(batch_tag)

    def build_model(self,layers=1,units=100,dropout=0.5):
        input_data = Input(shape=(self.seq_len,self.feature_len), name="input_layer") 
        y = input_data
        for a in range(layers):
            y = Bidirectional(LSTM(units, return_sequences=True, activation='tanh',recurrent_activation='sigmoid', name="lstm_layer_"+str(a)), name="bi_lstm_"+str(a))(y)
            y = Dropout(dropout)(y)
        y = TimeDistributed(Dense(len(self.tagsets)+1, name="dense_layer"), name="td_dense")(y)
        output_data = Activation('softmax', name="activation_layer")(y)

        self.model = Model([input_data], [output_data])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    def train_model(self,batch_size=32,epochs=3,verbose=1):


        # checkpoints
        filepath="weights_best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint] 

        for _ in range(epochs):
            train_data = make_generator(self.train_data_list)
            train_batch_gen = batch_generator(self.train_data, batch_size, self.seq_len, self.feature_len, 115)
            validation_data_gen = make_generator(self.val_data_list)
            val_batch_gen = batch_generator(self.validation_data_gen, batch_size, self.seq_len, self.feature_len, 115)
            model.fit_generator(train_data, 
                                epochs=1, 
                                steps_per_epoch = ceil(train_data_length/batch_size), 
                                validation_data = validation_data_gen, 
                                validation_steps= ceil(val_data_length/batch_size),
                                callbacks=callbacks_list,
                                shuffle=True,
                                verbose=verbose
                        )
        
    
    
    def loader(self,model_name):
        try:
            self.model.load_weights(model_name)
        except Exception as e:
            print(e)

    def predict_pos(self,sentence):
        '''
            params : sentence -> str 
        '''
        tokens = nepali_tokenizer.tokenize_words([sentence.strip()])
        stemmed_tokens = nepali_stemmer.stem_corpus([tokens])
        encoded_tokens = words_encode(tokens)
        predictions = self.model.predict(encoded_tokens)[0].argmax(axis=1)
        return [(word,ts_sc_num2tags[tag]) for word,tag in zip(stemmed_tokens[0],predictions[predictions>0])]
    

pos_tagger = POS()
pos_tagger.build_model(layers=2)
pos_tagger.loader('pos_model_9808_9742_.h5')
print(pos_tagger.predict_pos('डा. केसीले शुक्रबारदेखि बन्जरहाको स्वास्थ्य चौकीमा आएका विरामीहरुको स्वास्थ्य परीक्षण गरिरहेका छन् ।'))


# [('डा.', 'FB'), ('केसी', 'NP'), ('ले', 'IE'), ('शुक्रबार', 'NN'), ('देखि', 'II'), 
# ('बन्जरहा', 'VE'), ('को', 'IKM'), ('स्वास्थ्य', 'NN'), ('चौकी', 'NN'), ('मा', 'II'), 
# ('आ', 'FZ'), ('एका', 'MM'), ('विरामी', 'NN'), ('हरु', 'IH'), ('को', 'IKM'), 
# ('स्वास्थ्य', 'NN'), ('परीक्षण', 'NN'), ('गरिरह', 'RR'), ('ेका', 'VE'), ('', 'NN'), 
# ('छन्', 'VVYX2')]