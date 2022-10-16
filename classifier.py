import pandas as pd
import numpy as np
import os
import glob
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from ast import literal_eval

### No.4 in pipeline: prepare data for classifier & train

model_1600 = Word2Vec.load("models\\1600_word2vec.model")
#model_1700 = Word2Vec.load("aligned\\1700_word2vec.model")
#model_1700 = Word2Vec.load("aligned\\1800_word2vec.model")
dn = os.path.abspath('classifier.py')
parent_dir = os.path.join(os.path.dirname(dn),'corpora\processed')
data_dir = os.path.join(os.path.dirname(dn),'data')


# should be also used by classifyier to convert test data, so check if in vocab necessary
# sentences should be list of tokens (?)
def sentence_vec(sentence):
    vec_sentence = np.zeros(model_1600.vector_size) #all models have vector size of 100

    ##super weirdnes of pandas to_csv/read.csv adding quotation marks
    sentence = sentence.replace('[','')
    sentence = sentence.replace(']','')
    sentence = sentence.replace("'","")
    sentence = sentence.replace(" ","")
    sentence = sentence.split(",")
    for word in sentence:
        if word in model_1600.wv: #vocabulary of all models are the same after alignement
           # word_count +=1 # no ++ in Python?
            vec_sentence += model_1600.wv[word] #write/add word vector to sentence vector values (vector size stays the same)
    word_count=len(sentence)
    vec_sentence = vec_sentence/word_count # average value
    return vec_sentence

#print(sentence_vec(["Du", "bist", "heute", "ein", "Baum"]))
#print(sentence_vec("Du bist heute ein Baum"))

def sentence_vec_to_csv():
    for file in os.listdir(parent_dir):
        #if dir/file check or ONLY FILES ON THIS LEVEL
        child_path = os.path.join(parent_dir,file)
        df = pd.read_csv(child_path,sep=";")#converters={'tokenized': lambda x: x[1:-1].split(',')}) 
        df['sentence_vec'] = [sentence_vec(item) for item in df['tokenized']]
        df.to_csv(child_path,mode="w",index=False,sep=";")
        print(df.head())
        print(file+" is done")
        #does it make sense to store sentence vec in multi. columns so that each vector[i] can be compared?
#sentence_vec_to_csv()


files = os.path.join(parent_dir,"*_corpus_proc.csv")
files = glob.glob(files)

#prepare train test
'''df = pd.read_csv(os.path.join(parent_dir,"1600_corpus_proc.csv"),usecols=["sentence_vec"],header=0,sep=",")
print(df.head())
print(len(df))
print(len(df.columns))'''

def fix_this_shit(my_string):
    my_string=my_string.replace('[','')
    my_string=my_string.replace(']','')
    #my_string = my_string.split()
    #return my_string
    floats_list = []
    for item in my_string.split():
        floats_list.append(float(item))
    return floats_list

def train_this_model():
    vectors1 = pd.read_csv(os.path.join(parent_dir,"1600_corpus_proc.csv"),converters={"sentence_vec": fix_this_shit},header=0,sep=",")["sentence_vec"]
    vectors2 = pd.read_csv(os.path.join(parent_dir,"1700_corpus_proc.csv"),converters={"sentence_vec": fix_this_shit},header=0,sep=";")["sentence_vec"]
    vectors3 = pd.read_csv(os.path.join(parent_dir,"1800_corpus_proc.csv"),converters={"sentence_vec": fix_this_shit},header=0,sep=";")["sentence_vec"]
    vs=[vectors1,vectors2,vectors3]
    vectors = pd.concat(vs).to_list()
    #vectors = pd.concat([pd.read_csv(file,sep=";",usecols=["sentence_vec"])for file in files], ignore_index=True).to_list() #use once all have same seperator
    year1 = pd.read_csv(os.path.join(parent_dir,"1600_corpus_proc.csv"),header=0,sep=",")["year"]
    year2 = pd.read_csv(os.path.join(parent_dir,"1700_corpus_proc.csv"),header=0,sep=";")["year"]
    year3 = pd.read_csv(os.path.join(parent_dir,"1800_corpus_proc.csv"),header=0,sep=";")["year"]
    ys=[year1,year2,year3]
    year = pd.concat(ys).to_list()
    #year = pd.concat([pd.read_csv(file,sep=";",usecols=["year"])for file in files], ignore_index=True).to_list()
    vectors_train, vectors_test, year_train, year_test = train_test_split(vectors, year, test_size=0.2,random_state=1)
    train_df = pd.DataFrame({'vectors_train':vectors_train, 'year_train':year_train})
    test_df = pd.DataFrame({'vectors_test':vectors_test, 'year_test':year_test})
    train_df.to_csv(os.path.join(data_dir,"train_word_embeds.csv"),mode="w",index=False,sep=";")
    test_df.to_csv(os.path.join(data_dir,"test_word_embeds.csv"),mode="w",index=False,sep=";")

    #classifier
    classifier = LogisticRegression()
    classifier.fit(vectors_train,year_train)

    #save model
    filename = 'classifier_word_embeds.sav'
    pickle.dump(classifier, open(filename, 'wb'))
train_this_model() 
'''
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)'''




'''files = os.path.join(parent_dir,"*_corpus_proc.csv")
files = glob.glob(files)
def set_master_corpus():      
    df = pd.concat([pd.read_csv(file,sep=";")for file in files], ignore_index=True)  
    df.to_csv("master.csv",mode="w",index=False,sep=";")
set_master_corpus()'''



'''vecs = 
for i, v in enumerate(df['Text_vect_avg']):
    print(len(df['Text_Tokenized'].iloc[i]), len(v))'''