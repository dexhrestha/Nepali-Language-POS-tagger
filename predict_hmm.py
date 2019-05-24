from math import log
from train_hmm import start_tag
import pickle
import numpy as np

class PredictHMM():    
    def __init__(self):
        self.viterbi_probabilities ={}
        self.backpointers={} 

        self.dictionaries = {}

    def load(self):
        with open('hmmModel.pickle','rb') as f:
            self.dictionaries = pickle.load(f)

        self.transition = self.dictionaries["transmission"]
        self.emission = self.dictionaries["emission"]
        self.word2tag = self.dictionaries["word2tag"]
        self.bigram_counts = self.dictionaries["bigram"]
        self.unique_tags = self.dictionaries["unique_tags"]

        """ New probabilities """
        self.transition_backoff = self.dictionaries["transition_backoff"]
        self.emission_backoff = self.dictionaries["emission_backoff"]
        self.transition_singleton = self.dictionaries["transition_singleton"]
        self.emission_singleton = self.dictionaries["emission_singleton"]
        self.transition_one_count = self.dictionaries["transition_smoothed"]
        self.emission_smoothed = self.dictionaries["emission_smoothed"]
        self.tag_count = self.dictionaries["tag_count"]
        self.n = self.dictionaries["n"]
        self.transition_minimum = -100000
        self.c_2tag = self.dictionaries["c_2tag"]

    def _get_smoothed_emission(self,w,t):
        if (w,t) in self.emission_smoothed:
            return self.emission_smoothed[(w,t)]
        else:
            lamda = 1+self.emission_singleton.get(t,0)
            return log(float(lamda*self.emission_backoff[w])/float(lamda+self.tag_count[t]))
    
    def _get_smoothed_transition(self,s,u,v):
        if (s,u,v) in self.transition_one_count:
            return self.transition_one_count[s,u,v]
        else:
            lamda = 1+self.transition_singleton.get((s,u),0)
            return log(float(lamda*self.transition_backoff[v])/float(lamda+self.c_2tag[s][u]))
    
    def base_case(self,w,t):
        emission = self._get_smoothed_emission(w,t)
        transition = self._get_smoothed_transition(t,start_tag,start_tag)
        return emission + transition, transition
    
    def recover_tags(self,sentence):
        pos_tag_indices = []
        for j, word in reversed(list(enumerate(sentence))):
            if j == len(sentence) - 1:
                maxi = self.transition_minimum
                insert_value = None
                for tag1 in self.word2tag[word]:
                    for tag2 in self.word2tag[sentence[j - 1]]:
                        tag_tuple = (tag2, tag1)
                        if self.viterbi_probabilities[(tag_tuple, j)] > maxi:
                            maxi = self.viterbi_probabilities[(tag_tuple, j)]
                            insert_value = (word, tag_tuple)
                pos_tag_indices.insert(0, insert_value)
            else:
                pos_tag_indices.insert(0, (word, self.backpointers[(pos_tag_indices[0][1], j + 1)]))
        return [(tup[0], tup[1][1]) for tup in pos_tag_indices[1:]]
    
    
    def decode(self,sentence):
        """ loop through words """
        sentence.insert(0, start_tag)
        for j in range(1, len(sentence)):
            word = sentence[j]
            """ loop through possible POS tags """
            for tag_i in self.word2tag[word]:
                    
                    """ Base case needs to be handled here """
                    if j == 1:
                        
                        tag_tuple = (start_tag, tag_i)
                        viterbi, transition = self.base_case(word, tag_i)

                        """ # calculate score using P(Ci|'^', '^') and P(Wj|Ci) """
                        self.viterbi_probabilities[(tag_tuple, j)] = viterbi

                        """ initialize backpointer for this word to 0 """
                        self.backpointers[(tag_tuple, j)] = (start_tag,start_tag)
                        
                    else:
                        
                        max_viterbi = self.transition_minimum
                        backpointer = None

                        """ Emission probability P(w | tag_i) """
                        emission_probability = self._get_smoothed_emission(word, tag_i)

                        """ Loop over all possible pair of tags for the previous 2 words """
                        for tag_j in self.word2tag[sentence[j - 1]]:
                            tag_tuple = (tag_j, tag_i)  # TODO can be reduced to just the previous tag
                            """ All possible tags for current - 2th word """
                            for tag_k in self.word2tag[sentence[j - 2]]:
                                transition_probability = self._get_smoothed_transition(tag_k, tag_j, tag_i)
                                """ Viterbi Log probability """
                                
                                viterbi = self.viterbi_probabilities[((tag_k, tag_j), j - 1)] + transition_probability + emission_probability

                                """ Calculating the max and backpointer to recover the tag sequence """
                                if viterbi > max_viterbi:
                                    max_viterbi = viterbi
                                    backpointer = (tag_k, tag_j)

                            self.viterbi_probabilities[(tag_tuple, j)] = max_viterbi
                            self.backpointers[(tag_tuple, j)] = backpointer

        pos_tagging = self.recover_tags(sentence)
        print("Printing POS tags")
        
        word_tag = []
        for word, tag in pos_tagging:
                word_tag.append((word,tag))

        return word_tag

    def evaluate(self,test_sentences,test_tags):
        # metric = []
        predicted = []
        for sentence in test_sentences:
            wordtag = np.asarray(self.decode(sentence))[:,1]
            predicted.append(wordtag)
        # print(wordtag)
        acc = []
        for act_tags,pred_tags in zip(test_tags,predicted):
            tacc = []
            for act,pred in zip(act_tags,pred_tags):
                tacc.append(act==pred)
            acc.append(np.sum(tacc)/len(act_tags))
        
        
        self.acc = np.sum(acc)/len(test_sentences)
        print("Accuracy: ",self.acc)

        


if __name__ == '__main__':
    hmm = PredictHMM()
    hmm.load()
    print(hmm.evaluate([['बैंक' ,'र' ,'कम्पनी', 'बीच','सम्झौता']],[['NN','N','C','II','NN']]))
    # print(hmm.n)