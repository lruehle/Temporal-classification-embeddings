from gensim.utils import simple_preprocess
import os
from nltk.corpus import stopwords
import pandas as pd
import csv


### base version of convert_corpus
### if processing after model is still necessary

dn = os.path.abspath('tokenize_corpus.py')
parent_dir = os.path.join(os.path.dirname(dn),'corpora\processed')

stop_words = stopwords.words('german')


def remove_stop_words(txt):
    token = txt.split()
    return ' '.join([w for w in token if not w in stop_words])

def preproc(txt):
    txt = remove_stop_words(txt)
    return simple_preprocess(txt,deacc=True,min_len=3, max_len=20) #should return list of strings

for file in os.listdir(parent_dir):
    #if dir/file check or ONLY FILES ON THIS LEVEL
    child_path = os.path.join(parent_dir,file)
    df = pd.read_csv(child_path)
    #print(df.head())
    #df.drop('tokenized', axis=1, inplace=True)
    #df.drop('sentence_vec', axis=1, inplace=True)
    df['tokenized'] = df['txt'].map(preproc) 
    print(df.head())
    df.to_csv(child_path, mode="w",index=False,sep = ';')#,quoting = csv.QUOTE_NONE, escapechar = ' ')
    print(file+" is done")
    
    



