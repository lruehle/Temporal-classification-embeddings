from itertools import count
import pandas as pd
import numpy as np
import os
import glob
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier

### No.4 in pipeline: prepare data for classifier & train

model_1600 = Word2Vec.load("models_century\\1600_word2vec.model")
model_1700 = Word2Vec.load("aligned\century\\1700_word2vec.model")
model_1800 = Word2Vec.load("aligned\century\\1800_word2vec.model")
model_1600_erw = KeyedVectors.load("aligned\erw\\1600erw_word2vec.model", mmap='r')
model_1700_erw = KeyedVectors.load("aligned\erw\\1700erw_word2vec.model", mmap='r')
model_1800_erw = KeyedVectors.load("aligned\erw\\1800erw_word2vec.model", mmap='r')
#model_grimm = Word2Vec.load("aligned\grimm\\1800_word2vec.model")
#model_grimm = Word2Vec.load("aligned\grimm\\1800erw_word2vec.model")
dn = os.path.abspath('classifier.py')
parent_dir = os.path.join(os.path.dirname(dn),'corpora\processed\century')
data_dir = os.path.join(os.path.dirname(dn),'data')


def load_pickle(pickle_path):
    return pd.read_pickle(pickle_path)

def get_all_df(parent_dir):
    all_files = glob.glob(os.path.join(parent_dir, "*_corpus_proc.csv"))
    df = pd.concat((pd.read_csv(f,sep=";",header=None,names=["txt","year","tokenized"]) for f in all_files), ignore_index=True)
    return df


def year_distribution(data_f):
    #if df is None:
     #   df = get_all_df()
    print(data_f.groupby('year').size())
    data_f.groupby('year').size().plot(kind='bar')


# should be also used by classifyier to convert test data, so check if in vocab necessary
# sentences should be list of tokens (?)
def sentence_vec(sentence,year):
    vec_sentence = np.zeros(100)#model_1600.vector_size) #all models have vector size of 100
    ##super weirdnes of pandas to_csv/read.csv adding quotation marks
    sentence = sentence.replace('[','')
    sentence = sentence.replace(']','')
    sentence = sentence.replace("'","")
    sentence = sentence.replace(" ","")
    sentence = sentence.split(",")
    #model = model_1600 if year == 1600 else model_1700 if year == 1700 else model_1800
    model = model_1600_erw if year == 1600 else model_1700_erw if year == 1700 else model_1800_erw
    #model = model_grimm
    counter =0
    for word in sentence:
        if word in model.wv: #vocabulary of all models are the same after alignement
           # word_count +=1 # no ++ in Python?
            vec_sentence += model.wv[word] #write/add word vector to sentence vector values (vector size stays the same)
        else :
            counter +=1
    word_count=len(sentence)
    vec_sentence = vec_sentence/word_count # average value
    return vec_sentence

#sentences = dataframe
def sentences_to_vec(sentences,years):
    #sentence_vecs=pd.DataFrame()
    sentence_vex = list()
    counter = 0
    for idx,item in enumerate(sentences):
        #arr = sentence_vec(item)
        #print(arr)
        if (counter == 49999):
            print('Progress report:\n sentence to vec at 50000')
            counter = 0
        counter += 1
        year = years.iloc[idx]
        vecs = list(sentence_vec(item,year))
        sentence_vex.append(vecs)
        #for some reason pd stores vec in rows instead of cols??? extreme runtime improvement list.append vs. pd.concat 5h -> 2min
        '''vecs = pd.DataFrame(sentence_vec(item,year)).T
        #print(vecs.head())
        sentence_vecs = pd.concat([sentence_vecs,vecs],axis=0,ignore_index=True)'''
    print(sentence_vex[1])
    sentence_vecs= pd.DataFrame(sentence_vex)
    print(sentence_vecs.head())
    return sentence_vecs

def create_data_pickle(data_size=10000):
    all_df = get_all_df(parent_dir)
    all_df = all_df.groupby("year").sample(n=data_size, random_state=1)
    all_df = all_df.sample(frac=1).reset_index(drop=True) #shuffle values
    print(all_df.head())
    sentence_vecs = pd.DataFrame()
    sentence_vecs = sentences_to_vec(all_df['tokenized'],all_df['year'])
    #sentence_vecs = sentences_to_vec(all_df[:data_size]['tokenized']) no longer needed, as sample is taken now
    print(all_df.shape)
    print(all_df.shape[0])
    print(sentence_vecs.shape)
    sentence_vecs['year'] = all_df['year']
    print("\n vec head: ",sentence_vecs.head())
    sentence_vecs.to_pickle('data\centuries\master_vecs_200k.pkl')


#create train/test data:
#load & check
#! Change for Bayes!
def create_train_test():
    sentence_vecs = load_pickle('data\centuries\master_vecs_200k.pkl') 
    '''sentence_vecs_grimm = load_pickle('data\grimm\master_vecs_grimm_all.pkl')
    sentence_vecs_grimm['year'] = 1800.0
    frames= [sentence_vecs1, sentence_vecs_grimm]
    sentence_vecs = pd.concat(frames)'''
    #print("1 head: \n",sentence_vecs1.head(), sentence_vecs1.dtypes)
    #print("grimm head \n",sentence_vecs_grimm.head(),sentence_vecs_grimm.dtypes)
    print("creating train_test for: \n",sentence_vecs.head())
    #print("size 1: \n",sentence_vecs1.shape)
    print("size total: \n",sentence_vecs.shape)
    year_distribution(sentence_vecs)

    #training split:
    year =sentence_vecs.columns[-1]
    all_vecs =sentence_vecs.columns[:-1]
    X=sentence_vecs[all_vecs].values
    y=sentence_vecs[year].values

    # normalize for Naive Bayes (lots of negatives in the vectors)
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    #normalized_X = normalize(scaled_X, norm='l1', axis=1, copy=True)

    #X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.3, random_state=42) #Bayes, no difference between scaled/unscaled for Dtree 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print("amount training 1600|1700|1800",counts_train)
    print("amount testing 1600|1700|1800",counts_test)
    #check data
    print("training shape vectors: ",X_train.shape)
    print("training shape years: ",y_train.shape)
    print("testing shape vectors: ",X_test.shape)
    print("testing shape years: ",y_test.shape)
    #save data 
    # comb
    '''dump(X_train,"data\combined\X_train_200_comb.joblib")
    dump(X_test,"data\combined\X_test_200_comb.joblib")
    dump(y_train,"data\combined\y_train_200_comb.joblib")
    dump(y_test,"data\combined\y_test_200_comb.joblib")'''

    dump(X_train,"data\centuries\X_train_200.joblib")
    dump(X_test,"data\centuries\X_test_200.joblib")
    dump(y_train,"data\centuries\y_train_200.joblib")
    dump(y_test,"data\centuries\y_test_200.joblib")

#create_data_pickle(200000)
#create_data_pickle()
#create_train_test()

#load and print train/test
def create_classifier():
    #grimm + dta corpus:
    '''X_train = load("data\combined\X_train_200_comb.joblib")
    X_test=load("data\combined\X_test_200_comb.joblib")
    y_train=load("data\combined\y_train_200_comb.joblib")
    y_test=load("data\combined\y_test_200_comb.joblib") '''
    X_train = load("data\centuries\X_train_200.joblib")
    X_test=load("data\centuries\X_test_200.joblib")
    y_train=load("data\centuries\y_train_200.joblib")
    y_test=load("data\centuries\y_test_200.joblib")

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print("amount training 1600|1700|1800",counts_train)
    print("amount testing 1600|1700|1800",counts_test)

    # naive Bayes:
    '''classifier_nb = MultinomialNB()
    classifier_nb.fit(X_train,y_train)
    prediction=classifier_nb.predict(X_test)
    dump(classifier_nb, 'classifier\centuries\\nb_centuries_200.joblib')
    print("\n\nNaive Bayes Classifier:\n")'''
 

    #logistic Regression:
    #For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; #multi_class = auto -> multinomial
    '''classifier_logR = LogisticRegression(C=20.0,penalty='l2', solver='sag').fit(X_train,y_train) #c:regularization (trust this data alot/less values from 0.001 - 1k)
    prediction = classifier_logR.predict(X_test)
    dump(classifier_logR, 'classifier\combined\logR_centuries_200_comb.joblib')
    print("\n\nLogistic Regression Classifier:\n With values: c=20; penalty=L2, solver=saga:\n ")'''

    # Decisiontree:
    classifier_Dtree = DecisionTreeClassifier(max_depth=20,min_samples_leaf=1,criterion='gini') #check criterion, min_leaf=1 best for few classes
    classifier_Dtree = classifier_Dtree.fit(X_train,y_train)
    prediction = classifier_Dtree.predict(X_test)
    dump(classifier_Dtree, 'classifier\centuries\Dtree_centuries_200.joblib')
    print("\n\nDecision Tree Classifier:\n With values: depth=20; criterion=gini, min_samples_leaf=1\n ")

    #print classifier results:
    print("classification report: \n",metrics.classification_report(y_test, prediction))
    #print("confusion matrix for 1600, 1700, 1800 (x=true; y=predicted):\n",confusion_matrix(y_test,prediction))
    print(pd.crosstab(y_test, prediction, rownames=['True values'], colnames=['Predicted'], margins=True))

#create_classifier()

def classify_this(data_vecs,classifier,truthy=None):
    prediction=classifier.predict(data_vecs)
    print(prediction)
    if truthy is not None:
        print("classification report: \n",metrics.classification_report(truthy, prediction))
        print(pd.crosstab(truthy, prediction, rownames=['True values'], colnames=['Predicted'], margins=True))



#vector_df = pd.DataFrame()
def sentence_vec_to_csv():
    for file in os.listdir(parent_dir):
        #if dir/file check or ONLY FILES ON THIS LEVEL
        child_path = os.path.join(parent_dir,file)
        df = pd.read_csv(child_path,sep=";",header=None,names=["txt","year","tokenized"])#converters={'tokenized': lambda x: x[1:-1].split(',')}) 
        df['sentence_vec'] = [sentence_vec(item) for item in df['tokenized']]
        #vector_df['sentence_vec'] += df
        #df.drop('txt',axis=1)
        # df.drop('tokenized',axis=1)
        df.to_csv(child_path,mode="w",index=False,header=False,sep=";")
        print(df.head())
        print(file+" is done")
        #does it make sense to store sentence vec in multi. columns so that each vector[i] can be compared? ->yup! now pickle, 10x as fast
#sentence_vec_to_csv()




