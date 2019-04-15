'''
    Implementation of Stemmer in Nepali Language
    Authors: Dexter Shrestha
    Credits: Pravesh Koirala (Stemmer for Nepali Language-NASCOIT) 
'''
import os

# relative paths
suf_i = os.path.join(os.path.dirname(__file__), 'utils/suffixI.txt')
suf_ii = os.path.join(os.path.dirname(__file__), 'utils/suffixII.txt')
exception_i = os.path.join(os.path.dirname(__file__), 'utils/exceptionI.txt')

class Stemmer():

	def __init__(self):
		#get suffix I
		with open(suf_i,'r',encoding='utf-8') as f:
			suffixI = f.read()
		self.suffixI = suffixI.strip().split('\n')

		#get suffix II
		with open(suf_ii,'r',encoding='utf-8') as f:
			suffixII = f.read()
		self.suffixII = suffixII.strip().split('\n')
		self.suffixII = [x.strip() for x in self.suffixII]

		#get exception I
		with open(exception_i,'r',encoding='utf-8') as f:
			execptionI = f.read()
		self.exceptionI = execptionI.strip().split('\n')
		
	def remove_suffixI(self,word):
		if word in self.exceptionI:
			return word
		elif word in self.suffixI:
			return word
		else:
			tempSuffix = []
			for x in self.suffixI:
			    if x in word:
			        tempSuffix.append(x)
			#print(tempSuffix)
			if len(tempSuffix) > 1:
			    #print('more suff')
			    return self.iterativeStem(tempSuffix,word)
			elif len(tempSuffix) == 1:
			    #print('single suff')
			    return word[:-len(tempSuffix[0])]
			else:
			    #print('no suff')
			    return word
			    
	def iterativeStem(self,suffList,word):
		if len(word) > 2: 
			for x in range(len(suffList)):
				if not word.endswith(suffList[x]) and x==len(suffList[x])-1:
						#print('stem2')
						return word
				elif word.endswith(suffList[x]):
					#print('stem1')
					word = word[:-len(suffList[x])]
					return self.iterativeStem(suffList,word)
				else: 
					continue
			return word
		return word

	def normalize(self,word):
		normRules = {
		'ई' : 'इ',
		'◌ी':'ि',
		'ऊ':'उ',
		'ू':'ु',
		'व':'ब',
		'श':'स',
		'ष':'स',
		'ँ':''
	}
		#normalize the word
		tstr=''
		wlist = [x for x in word]
		for x in wlist:
			tstr = ''
			if x in normRules.keys():
				#print('inside')
				i = word.index(x)
				wlist[i] = normRules[x]
				for a in wlist:
				    tstr = tstr+a
				return self.normalize(tstr)
		if tstr == '':
			return word
		else:
			return tstr

	def remove_suffixII(self,word):
		if word in self.suffixII:
			return word
		word = self.normalize(word)
		tSuff = []
		probab = []
		for x in self.suffixII:
			if x in word:
			    tSuff.append(x)
		#print(tSuff) #interested candidate
		for x in tSuff:
			if word.endswith(x):
			    #needs to be fixed ... try different approaches
	#             i = word.find(x)-len(x)-1
			    d = len(word)-word.find(x)
	##WHy??? - > ans :
	#d = len(word)-word.find('हु')
	#word[:-d-len('हु')]
			    if len(word[:-d-(len(x)-1)])  >= 2 :
			      #  print('stripped')
			        #print(x)
			        tword = word[:-d-(len(x)-1)]
			        probab.append(tword)
	#                 print(probab)
			    else:
			        tSuff.remove(x)
			        continue
			    if tword == None or tword == word:
			        return word
		try:
			#print(probab)
	#         l = probab[0]
	#         for x in probab:
	#             if len(x) < l:
	#                 l = len[x]
	#         return l
			return probab[0]
		except:
			return word

	def stem_word(self,word):
		word=word.strip()
		tword = self.remove_suffixI(word)
		if tword == word:
			#print(tword)
			word = self.remove_suffixII(tword)
			return word
		return tword

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
		# print(t_sentence)
		s = []
		for word in t_sentence:
			s.append(self.stem_word(word))
		return s
		# return self.stem_word(t_sentence)
