import pandas as pd

def csv_to_corpus(filename,start_tag):
    df = pd.read_csv(filename)
    df = df[["tags","words"]]
    corpus = []
    sent = []
    for tag,word in zip(df["tags"],df["words"]):
        if(tag != '.'):
            sent.append((word,tag))
        else:
            corpus.append(sent)
            sent = []

    def insert_start(sentence,tag):
        sentence.insert(0,(tag, tag))
        sentence.insert(0,(tag, tag))
        return sentence

    corpus = list(map(lambda x:insert_start(x,start_tag),corpus))
    
    return corpus

