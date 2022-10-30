import convert_corpus
import os
import embeddings
import align
import classifier
from sklearn.preprocessing import MinMaxScaler, normalize
from joblib import load

# preprocessing and classifying new texts
#all you have to do is change paths 
dn = os.path.abspath('test_grimm.py')
output_src = os.path.join(os.path.dirname(dn),'corpora\processed')
csv_data = os.path.join(output_src,'1700_erw_corpus_skip_proc.csv')
model_path = os.path.join(os.path.dirname(dn),"aligned\skip_grimm\\1800grimm_skip.model")

#models

'''model_grimm = align.load_model("models_skipgram\\grimm_1800_skipgramm.model")
#model_grimm_aligned = align.load_model("aligned\grimm\\1800erw_word2vec.model")
model1 = align.load_model("aligned\skip\\1600_skip_align.model")
model2 = align.load_model("aligned\skip\\1700_skip_align.model")
model3 = align.load_model("aligned\skip\\1800_skip_align.model")
skip_1 = align.load_model("models_skipgram\\erw_1600_skipgramm.model")
skip_2 = align.load_model("models_skipgram\\erw_1700_skipgramm.model")
skip_3 = align.load_model("models_skipgram\\erw_1800_skipgramm.model")'''


#classifiers:
classifier_nb = load('classifier\cbow\\nb_cbow.joblib')
classifier_nb_skip = load('classifier\skip\\nb_skip.joblib')
# classifier_b_nb = load('classifier\centuries\\nb_centuries_b.joblib')
classifier_logR = load('classifier\cbow\\logR_cbow.joblib')
classifier_logR_skip = load('classifier\skip\\logR_skip.joblib')
# classifier_b_logR = load('classifier\centuries\\logR_centuries_b.joblib')
# classifier_combined_logR =load('classifier\combined\\logR_centuries_200_comb.joblib')

classifier_dTree = load('classifier\cbow\\Dtree_cbow.joblib')
classifier_dTree_skip = load('classifier\skip\\dTree_skip.joblib')
# classifier_b_dTree = load('classifier\centuries\\Dtree_centuries_b.joblib')




# create one file out of all, preprocess and add year & token
# hier eigentlich kein year mehr..vllt platzhalter?
def process_grimm():
    parent_dir = os.path.join(os.path.dirname(dn),'corpora\erw') # make args
    #dir name has to be year
    for dir in os.listdir(parent_dir):
        #if dir/file check or NO FILES ON THIS LEVEL
        child_dir = os.path.join(parent_dir,dir)
        for file in os.listdir(child_dir):
            data =""
            file_path = f"{child_dir}\{file}"
            with open(file_path,"r",encoding="utf8") as f:
                for line in f:
                    if not line.isspace():
                        data += line
            data = data.replace(".",".\n")
            data = data.replace("\n\n","\n")
            with open(file_path,"w",encoding="utf8") as f:
                f.write(data)
        convert_corpus.proc_files_in_dir(child_dir, output_src, dir)
#process_grimm()

def create_grimm_emb():
    embeddings.create_embedding(csv_data)

# create_grimm_emb()

def align_grimm():
    align.align_models(model3, model_grimm, model_path)

#align_grimm()

def tokens_to_vec():
    df = classifier.get_all_df(output_src)
    #df = df.groupby("year").sample(n=6500, random_state=1)
    #df = df.sample(frac=1).reset_index(drop=True) 
    df = df.sample(frac=1).reset_index(drop=True) #randomize
    print(df.head())
    ### add grimm version to classifier!
    sentence_vecs = classifier.sentences_to_vec(df['tokenized'],df['year'])
    '''
    sentence_vecs_18 = classifier.sentences_to_vec(df['tokenized'],df['year'])
    df.loc[df['year'] == 1800, 'year'] = 1700
    sentence_vecs_17 = classifier.sentences_to_vec(df['tokenized'],df['year'])
    df.loc[df['year'] == 1700, 'year'] = 1600
    sentence_vecs_16 = classifier.sentences_to_vec(df['tokenized'],df['year'])
    sentence_vecs = (sentence_vecs_16 + sentence_vecs_17 + sentence_vecs_18).div(3)'''
    sentence_vecs['year'] = df['year']
    print(sentence_vecs.shape)
    sentence_vecs.to_pickle('data\skip_erw\master_vecs_skip_erw.pkl')

def get_classified():
    # sentence_vecs = classifier.load_pickle('data\erw\master_vecs_erw.pkl') 

    sentence_vecs = classifier.load_pickle('data\cbow_erw\master_vecs_cbow_erw.pkl')
    all_vecs =sentence_vecs.columns[:-1]
    X=sentence_vecs[all_vecs].values
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    # normalized_X = normalize(scaled_X, norm='l1', axis=1, copy=True)
    #print("your vectors are: \n",sentence_vecs.head())

    classifier.classify_this(scaled_X,classifier_nb, sentence_vecs['year'])#for nb restructure vectors for negative values
    print("skip Naive Bayes (200k) on erw-data: \n")
    # print("LogR (200k,c1,l2,sag) on erw-data: \n")

    # print("Decision Tree (200k,l5,d20,logloss) on erw-data: \n")
    #classifier.classify_this(sentence_vecs,classifier_nb)#for nb restructure vectors for negative values

get_classified()
#tokens_to_vec()
