import re
import os
'''
    Implementation of Tokenizer in Nepali Language
    Authors: Dexter Shrestha, Dipesh Dulal
    Credits: http://www.nepalinlp.com/detail/nepali-texts-tokenization/ 
'''

class Tokenizer:
    def __init__(self):
        self.words = []
        stop_word_file = os.path.join(os.path.dirname(__file__), 'no_stop_words.txt')
        with open(stop_word_file, 'r') as f:
            self.stop_words = f.read().splitlines()

    # private tokenize_sentence method
    def __tokenize_sentence(self,sentence):
        return re.split('(?<=[।?!]) +', sentence)
    
    # @param corpus [to feed in a large article corpus]
    # and return tokenized corpus in the format
    # [ [ [w1], [w2], ... [wn] ], 
    #   [ [w1], [w2], ... [wn] ], ... ]
    # wn represents words in a sentence
    # corpus in format [ [a1], [a2], [a3], ... [an] ]
    # an represents articles
    def tokenize_corpus(self, corpus):
        # tokenizing sentences
        sentences = []
        for article in corpus:
            sentences_ = self.__tokenize_sentence(article)
            for sentence in sentences_:
                sentences.append(sentence)

        # tokenizing words
        tokenized_corpus = []
        for sentence in sentences:
            tokenized_corpus.append(self.tokenize_words([sentence]))
        
        return tokenized_corpus

    def __remove_stop_words(self, words):
        r_words = []
        for word in words:
            if word not in self.stop_words:
                r_words.append(word)
        return r_words

    def tokenize_words(self,sentences):
        # need to add colon lexicon
        colon_lexicon = ['अंशत:', 'मूलत:', 'सर्वत:', 'प्रथमत:', 'सम्भवत:', 'सामान्यत:', 'विशेषत:', 'प्रत्यक्षत:',
        'मुख्यत:', 'स्वरुपत:', 'अन्तत:', 'पूर्णत:', 'फलत:', 'क्रमश:', 'अक्षरश:', 'प्रायश:',
        'कोटिश:', 'शतश:', 'शब्दश:']

        words=[]
        for sentence in sentences:
            sentence = re.sub('\,|\"|\'| \)|\(|\)| \{| \}| \[| \]|!|‘|’|“|”| \:-|\?|।|/|\—',
                          ' ',
                          sentence)
            sentence=sentence.split()
            for word in sentence:
                words.append(word)

        for word in words:
            if word[len(word) - 1:] == '-':
                words.append(word[:len(word) - 1])
                words.remove(word)

        for word in words:
            if word[len(word) - 1:] == ':' and word not in colon_lexicon:
                self.words.append(word[:len(word) - 1])
            else:
                self.words.append(word)

        return self.__remove_stop_words(words)
        



