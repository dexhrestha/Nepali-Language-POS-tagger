**Nepali Language POS tagger**  

  Dataset used: National Nepali Corpus

TO RUN:
```sh
$ pip install gensim
$ jupyter notebook
```

- `nlp_processor` contains stemming and tokenization library
- `stemmer_tokenizer_word2vec` contains the word2vec model for preprocessing

train_hmm.py 
- Calculates all requrired probabiliites form given corpus
```sh
$ python train_hmm  --corpus path/to/corpus
```

predict_hmm.py
- Predicts POS tag for a given sentence using decode function

```sh
$ python predict_hmm.py 
```

Dipesh Dulal, Dexter Shrestha
