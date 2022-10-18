import pandas as pd
import numpy as np
import os
import glob
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
from ast import literal_eval
from sklearn.tree import DecisionTreeClassifier

### No.4 in pipeline: prepare data for classifier & train

model_1600 = Word2Vec.load("models\\1600_word2vec.model")
#model_1700 = Word2Vec.load("aligned\\1700_word2vec.model")
#model_1700 = Word2Vec.load("aligned\\1800_word2vec.model")
dn = os.path.abspath('classifier.py')
parent_dir = os.path.join(os.path.dirname(dn),'corpora\processed')
data_dir = os.path.join(os.path.dirname(dn),'data')

def get_all_df():
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
def sentence_vec(sentence):
    vec_sentence = np.zeros(100)#model_1600.vector_size) #all models have vector size of 100

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

#sentences = dataframe
def sentences_to_vec(sentences):
    sentence_vecs=pd.DataFrame()
    for item in sentences:
        #arr = sentence_vec(item)
        #print(arr)
        #for some reason pd stores vec in rows instead of cols???
        vecs = pd.DataFrame(sentence_vec(item)).T
        #print(vecs.head())
        sentence_vecs = pd.concat([sentence_vecs,vecs],axis=0,ignore_index=True)
    print(sentence_vecs.head())
    return sentence_vecs

def create_data_pickle(data_size=10000):
    all_df = get_all_df()
    all_df = all_df.groupby("year").sample(n=data_size, random_state=1)
    all_df = all_df.sample(frac=1).reset_index(drop=True) #shuffle values
    print(all_df.head())
    sentence_vecs = pd.DataFrame()
    sentence_vecs = sentences_to_vec(all_df['tokenized'])
    #sentence_vecs = sentences_to_vec(all_df[:data_size]['tokenized']) no longer needed, as sample is taken now
    print(all_df.shape)
    print(all_df.shape[0])
    print(sentence_vecs.shape)
    sentence_vecs['year'] = all_df['year']
    print(sentence_vecs.head())
    sentence_vecs.to_pickle('master_vecs.pkl')
#create_data_pickle()

def load_pickle(pickle_path):
    return pd.read_pickle(pickle_path)

#load & check
sentence_vecs = load_pickle('master_vecs.pkl') 
print(sentence_vecs.head())
year_distribution(sentence_vecs)

#training split:
year =sentence_vecs.columns[-1]
all_vecs =sentence_vecs.columns[:-1]
X=sentence_vecs[all_vecs].values
y=sentence_vecs[year].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)

#check data
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# naive Bayes:


vector_df = pd.DataFrame()
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
        #does it make sense to store sentence vec in multi. columns so that each vector[i] can be compared?
#sentence_vec_to_csv()


#files = os.path.join(parent_dir,"*_corpus_proc.csv")
#files = glob.glob(files)

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
    return floats_list #[float1,float2...]

def train_this_model():
    vectors1 = pd.read_csv(os.path.join(parent_dir,"1600_corpus_proc.csv"),converters={3: fix_this_shit},sep=";", header=None)[3]
    #print(vectors1.head())
    #vectors1 = vectors1.to_list()
    #vectors1=[float(x) for inner_list in vectors1 for inner in inner_list for x in inner]
    #vectors1 = vectors1.astype(float)
    vectors2 = pd.read_csv(os.path.join(parent_dir,"1700_corpus_proc.csv"),converters={3: fix_this_shit},sep=";", header=None)[3]
    vectors3 = pd.read_csv(os.path.join(parent_dir,"1800_corpus_proc.csv"),converters={3: fix_this_shit},sep=";", header=None)[3]
    vs=[vectors1,vectors2,vectors3]
    vectors = pd.concat(vs).values.tolist() #needs to be list of float, not string! -> [[float1,float2],[float1,float2]]
    #v= np.fromstring(vectors,dtype=float)
    #vectors = pd.concat([pd.read_csv(file,sep=";",usecols=["sentence_vec"])for file in files], ignore_index=True).to_list() #use once all have same seperator
    year1 = pd.read_csv(os.path.join(parent_dir,"1600_corpus_proc.csv"),sep=";",header=None,usecols=[1])#["year"]
    #year1['year'] = year1['year'].astype(int)
    #year1 = year1.values.tolist()
    #year1 = [int(x) for inner_list in year1 for x in inner_list]
    #year1 = [float(x) for x in year1]
    year2 = pd.read_csv(os.path.join(parent_dir,"1700_corpus_proc.csv"),sep=";", header=None,usecols=[1])
    year3 = pd.read_csv(os.path.join(parent_dir,"1800_corpus_proc.csv"),sep=";", header=None,usecols=[1])
    ys=[year1,year2,year3]
    year = pd.concat(ys).values.tolist()
    #year = pd.concat([pd.read_csv(file,sep=";",usecols=["year"])for file in files], ignore_index=True).to_list()
    vectors_train, vectors_test, year_train, year_test = train_test_split(vectors, year, test_size=0.2,random_state=1)
    dtree_model = DecisionTreeClassifier(max_depth = 2).fit(vectors_train, year_train)
    dtree_predictions = dtree_model.predict(vectors_test)
    cm = confusion_matrix(year_test, dtree_predictions)
    print(cm)
    print('accuracy %s' % accuracy_score(dtree_predictions, year_test))
    print(classification_report(year_test, dtree_predictions,target_names=['1600','1700','1800']))
  #  train_array = np.array([vectors_train,year_train])
   # test_array = np.array([vectors_test, year_test])
#    np.save(os.path.join(data_dir,"train_word_embeds.npy"),train_array)
 #   np.save(os.path.join(data_dir,"test_word_embeds.npy"),test_array)

    #train_df = pd.DataFrame({'vectors_train':vectors_train, 'year_train':year_train})
    #test_df = pd.DataFrame({'vectors_test':vectors_test, 'year_test':year_test})
    #train_df.to_csv(os.path.join(data_dir,"train_word_embeds.csv"),mode="w",index=False,sep=";")
   # test_df.to_csv(os.path.join(data_dir,"test_word_embeds.csv"),mode="w",index=False,sep=";")
    #np.savetxt(os.path.join(data_dir,"train_word_embeds.txt"), train_df.values, delimiter=";",fmt='%s')
    #np.savetxt(os.path.join(data_dir,"test_word_embeds.txt"), test_df.values, delimiter=";",fmt='%s')

    #classifier
    #classifier = LogisticRegression()
    #classifier.fit(vectors_train,year_train)

    #save model
    #filename = 'classifier_word_embeds.sav'
    #pickle.dump(classifier, open(filename, 'wb'))
    #predicted = classifier.predict(vectors_test) #vector Übergabe -> für workflow muss eingabe noch in vector verarbeitet werden (sentence_vec())
    #print("Logistic Regression Accuracy:",metrics.accuracy_score(year_test, predicted))
    #print("Logistic Regression Precision:",metrics.precision_score(year_test, predicted))
    #print("Logistic Regression Recall:",metrics.recall_score(year_test, predicted))
    #result = classifier.score(vectors_test, year_test)
    #print(result)


#train_this_model() 

# load the model from disk
'''train_df = pd.read_csv(os.path.join(data_dir,"test_word_embeds.csv"),sep=";")
train_df = np.loadtxt(os.path.join(data_dir,"test_word_embeds.csv"),delimiter=";",skiprows=1)
print(train_df.flat[0])
vec_test = train_df['vectors_test'].to_list()
vec_test = [item[0] for item in train_df]
year_test = train_df['year_test'].to_list()
year_test = [item[1] for item in train_df]'''
'''all_test = np.load(os.path.join(data_dir,"test_word_embeds.npy"))
vec_test = [item[0] for item in all_test]
year_test = [item[1] for item in all_test]
loaded_model = pickle.load(open("classifier_word_embeds.sav", 'rb'))
predicted = loaded_model.predict(vec_test) #vector Übergabe -> für workflow muss eingabe noch in vector verarbeitet werden (sentence_vec())
print("Logistic Regression Accuracy:",metrics.accuracy_score(year_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(year_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(year_test, predicted))'''
#result = loaded_model.score(X_test, Y_test)
#print(result)




'''files = os.path.join(parent_dir,"*_corpus_proc.csv")
files = glob.glob(files)
def set_master_corpus():      
    df = pd.concat([pd.read_csv(file,sep=";")for file in files], ignore_index=True)  
    df.to_csv("master.csv",mode="w",index=False,sep=";")
set_master_corpus()'''


'''vecs = 
for i, v in enumerate(df['Text_vect_avg']):
    print(len(df['Text_Tokenized'].iloc[i]), len(v))'''