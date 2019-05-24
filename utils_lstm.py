import pandas as pd
import argparse,sys
from collections import Counter
import pickle
from keras.utils import to_categorical

def csv_to_corpus(filename):
    sentences = []
    sentence_tags = []
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(e)
        sys.exit()

    if(df.empty!=True):
        df = df[["tags","words"]]
        corpus = []
        sentence = []
        for tag,word in zip(df["tags"],df["words"]):
            if(tag!='.'):
                sentence.append((tag,word))
            else:
                corpus.append(sentence)
                sentence = []

        for sentence in corpus:
            x=[]
            y=[]
            for word in sentence:
                x.append(word[1])
                y.append(word[0])
            if len(x) > 0 and len(x)<200:
                sentences.append(x)
                sentence_tags.append(y)
        return sentences,sentence_tags
    return None
            
def map_tag2index(sentence_tags,output_file):
    labels = []
    for sentence in sentence_tags:
        labels.extend(sentence)
    labels = Counter(labels)
    tag2index = {t: i + 1 for i, t in enumerate(list(labels))}
    tag2index['-PAD-'] = 0

    with open(output_file,'wb') as f:
        pickle.dump(tag2index,f)

    return tag2index

def convert_tag2index(tag2index,sentence_tags):
    def tagsent2int(sent_tag):
        return [tag2index[tag] for tag in sent_tag]
    return list(map(tagsent2int,sentence_tags))

def  data_generator(sentences,sentence_tags,batch_size=3200):
    num_samples = len(sentences)
    print(num_samples)
    while(True):
        for offset in range(0,num_samples,batch_size):
            batch_sentences = sentences[offset:offset+batch_size]
            batch_sentence_tags = sentence_tags[offset:offset+batch_size]
            batch_sentence_tags=to_categorical(batch_sentence_tags,num_classes=109)
            # print(shape)
            yield(batch_sentences,batch_sentence_tags)


def logits_to_sentence(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
#             try:
            token_sequence.append(index[categorical])

        token_sequences.append(token_sequence)
    return token_sequences[0]