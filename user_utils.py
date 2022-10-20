# 
# preprocess input (file vs console)

import pandas as pd
import os
from smart_open import open
from nltk.corpus import stopwords
from pathlib import Path
import convert_corpus

# { 
# a: 
# # create embeddings? (based on file size?)
# # align embeddings 
# # convert sentences to vectors
# b:
# # convert sentences to vectors with other class
# # maybe have one embedding with vectors from all texts?
# }


## ask for file location
def proc_file(file):
    if file.endswith(".txt"): 
        df= df[df['txt'].str.count(' ') > 2] #drop lines with only one or two words -> references to role/active speaker etc.
        df['txt'] = df['txt'].apply(convert_corpus.remove_stop_words)
        # df = df. lemmatize me
        df = convert_corpus.remove_umlauts(df)
        df['tokenized'] = df['txt'].map(convert_corpus.preproc)
        df.to_csv(os.path.join(Path(file).stem+'_proc.csv'),mode="a",index=False,sep=";")
    else:
        print("can only process txt file so far")

#tokenize with 1600 model
