from collections import Counter
from collections import defaultdict
from utils_hmm import csv_to_corpus
from math import log

import numpy as np
import sys
import argparse
import nltk
import pickle

argp = argparse.ArgumentParser()

argp.add_argument("--corpus","-c",help="path to corpus file")


args = argp.parse_args()
start_tag = '-0-'

class POSHMM():

    def __init__(self,filename):
        self.filename = filename
        # self.filename='../major8sem/Dataset/nepali-cri-cn.csv'
        self.corpus = []
        self.dictionaries = {}
        self.words = []
        self.wordcount = Counter()
        self.tokenTags = defaultdict(Counter)
        self.N = 0
        self.tagcount=Counter()
        self.bigram_counts = {}
        self.trigram_counts = {}
        self.transition_singleton = {}
        self.emission_singleton = {}
        self.transition_backoff = {}
        self.emission_backoff= {}
        self.emission_smoothed={}
        self.transition_one_count={}
        self.word_given_pos={}
        self.pos3_given_pos2_and_pos1={}
        self.word_tag = {}

    def process_corpus(self):
        print("Loading corpus...")
        #process corpus from csv
        self.corpus = csv_to_corpus(self.filename,start_tag)
        #divide corpus to sentences
        for sentence in self.corpus:
            self.words.extend(sentence) 
        #get total number of word tag pair sepereated as sentences
        self.N = len(self.words)
        #get count of unique words and tags
        tokens,tags = zip(*self.words)
        self.wordcount = Counter(tokens)
        self.tagcount=Counter(tags)
        #number of words for a tag
        for token,tag in self.words:
            self.tokenTags[token][tag]+=1
        
        #word to tag mapping
        for word,tag in self.words:
            if word not in self.word_tag:
                self.word_tag[word] =set()
            self.word_tag[word].add(tag)
        

    def calculate_ngram_counts(self):
        print("Calculate Bigram and Trigram counts...")
        c_3tag = defaultdict(Counter)
        c_2tag = defaultdict(Counter)
        for sentence in self.corpus:
            tags= []
            for word,tag in sentence:
                tags.append(tag)
            trigram = nltk.ngrams(tags,3)
            bigram = nltk.ngrams(tags,2)
            for tag1,tag2,tag3 in trigram:
                c_3tag[(tag1,tag2)][tag3]+=1
            for tag1,tag2 in bigram:
                c_2tag[tag1][tag2]+=1
        for tag1 in c_2tag.keys():
            for tag2 in c_2tag[tag1].keys():
                tag = [tag1]
                tag.append(tag2)
                tag=tuple(tag)
                self.bigram_counts[tag] = c_2tag[tag1][tag2]

        
        for tag1 in c_3tag.keys():
            for tag2 in c_3tag[tag1].keys():
                tag = list(tag1)
                tag.append(tag2)
                tag=tuple(tag)
                self.trigram_counts[tag] = c_3tag[tag1][tag2]
        self.dictionaries['c_2tag'] = c_2tag
        self.dictionaries['c_3tag'] = c_3tag

    def smooth_trans(self,u, v, s):
        lamda = 1+self.transition_singleton.get((u,v),0) 
        return log(float(self.trigram_counts[(u,v,s)]+lamda*self.transition_backoff[s])/float(lamda+self.dictionaries['c_2tag'][u][v]))
            
    def smooth_emission(self,x,s):
        lamda = 1+self.emission_singleton.get(s,0)
        return log(float(self.tokenTags[x][s]+lamda*self.emission_backoff[x])/float(lamda+self.tagcount[s]))

    def trans(self,u,v,s):
        return log(float(1+self.trigram_counts[(u,v,s)])/float(len(list(self.tagcount.keys()))+self.bigram_counts[(u,v)]))

    def emission(self,x,s):
        return log(float(self.tokenTags[x][s])/float(self.tagcount[s]))
    
        

    def calculate_singleton(self):
        print("Calculate singleton probabilities")
        for i,tag1 in enumerate(self.tagcount.keys()):
            for j,tag2 in enumerate(self.tagcount.keys()):
                for k,tag3 in enumerate( enumerate(self.tagcount.keys())):
                    if i != j and i != k and j != k:
                            triplet = (tag3, tag2, tag1)
                            if triplet in self.trigram_counts and self.trigram_counts[triplet] ==1:
                                self.transition_singleton[(tag3, tag2)] = self.transition_singleton.get((tag3, tag2), 0) + 1
                                
        
        for word in  self.wordcount.keys():
            for tag in  self.tagcount.keys():
                if word in self.tokenTags.keys():
                    if self.tokenTags[word][tag] ==1 :                
                        self.emission_singleton[tag] = self.emission_singleton.get(tag, 0) + 1

    def calculate_backoff(self):
        print("Calculate backoff probabilities")        
        for word in self.wordcount.keys():
            self.emission_backoff[word] = float(1+self.wordcount[word])/float(self.N+len(list(self.tagcount.keys())))
        
        for tag in self.tagcount.keys():
            self.transition_backoff[tag] = float(self.tagcount[tag])/self.N
        
    def calculate_smoothed(self):
        print("Calculate smoothed probabilities")        
        for word,tag in self.words:
            self.emission_smoothed[(word,tag)]=self.smooth_emission(word, tag)

    def calculate_probabilities(self):
        print("Calculate conditional probabilities")
        for tag1,tag2,tag3 in self.trigram_counts:
            self.transition_one_count[(tag1,tag2,tag3)]=self.smooth_trans(tag1, tag2, tag3)
        
        for word,tag in self.words:
            self.word_given_pos[(word,tag)]=self.emission(word, tag)
        
        for tag1,tag2,tag3 in self.trigram_counts:
            self.pos3_given_pos2_and_pos1[(tag1,tag2,tag3)]=self.trans(tag1, tag2, tag3)

    def populate_dictionary(self):
        print("Save to dictionary")
        self.dictionaries['unique_tags'] = list(self.tagcount.keys())
        self.dictionaries['bigram'] = self.bigram_counts
        self.dictionaries['trigram'] = self.trigram_counts
        self.dictionaries['transition_singleton']= self.transition_singleton
        self.dictionaries['emission_singleton']= self.emission_singleton
        self.dictionaries['emission_backoff']= self.emission_backoff
        self.dictionaries["transition_backoff"]= self.transition_backoff
        self.dictionaries['emission_smoothed']= self.emission_smoothed
        self.dictionaries["transition_smoothed"] = self.transition_one_count
        self.dictionaries['emission']= self.word_given_pos
        self.dictionaries["transmission"] = self.pos3_given_pos2_and_pos1
        self.dictionaries['word2tag'] = self.word_tag
        self.dictionaries['tag_count']  = self.tagcount
        self.dictionaries['n'] = self.N

if __name__ == "__main__":
    hmm=POSHMM(args.__getattribute__('corpus'))
    hmm.process_corpus()
    hmm.calculate_ngram_counts()
    hmm.calculate_singleton()
    hmm.calculate_backoff()
    hmm.calculate_smoothed()
    hmm.calculate_probabilities()
    hmm.populate_dictionary()
    # print(hmm.corpus)
    # print(hmm.dictionaries['transmission'])

    with open('hmmModel.pickle','wb') as f:
        pickle.dump(hmm.dictionaries,f)
