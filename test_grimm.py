import convert_corpus
import os
import embeddings
import align
import classifier
from sklearn.preprocessing import MinMaxScaler, normalize
from joblib import load

#all you have to do is change paths 
dn = os.path.abspath('test_grimm.py')
output_src = os.path.join(os.path.dirname(dn),'corpora\processed\grimm')
csv_data = os.path.join(output_src,'1800_corpus_proc.csv')
model_path = os.path.join(os.path.dirname(dn),"aligned\grimm\\1800erw_word2vec.model")

#models
model_grimm = align.load_model("models_grimm\\1800erw_word2vec.model")
model_grimm_aligned = align.load_model("aligned\grimm\\1800erw_word2vec.model")
model3 = align.load_model("aligned\century\\1800_word2vec.model")

#classifiers:
classifier_nb = load('classifier\centuries\\nb_centuries_200.joblib')
classifier_logR = load('classifier\centuries\\logR_centuries_200.joblib')
classifier_dTree = load('classifier\centuries\\Dtree_centuries_200.joblib')



# create one file out of all, preprocess and add year & token
# hier eigentlich kein year mehr..vllt platzhalter?
def process_grimm():
    parent_dir = os.path.join(os.path.dirname(dn),'corpora\Laudatio') # make args
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
    ### add grimm version to classifier!
    sentence_vecs = classifier.sentences_to_vec(df['tokenized'],df['year'])
    '''
    sentence_vecs_18 = classifier.sentences_to_vec(df['tokenized'],df['year'])
    df.loc[df['year'] == 1800, 'year'] = 1700
    sentence_vecs_17 = classifier.sentences_to_vec(df['tokenized'],df['year'])
    df.loc[df['year'] == 1700, 'year'] = 1600
    sentence_vecs_16 = classifier.sentences_to_vec(df['tokenized'],df['year'])
    sentence_vecs = (sentence_vecs_16 + sentence_vecs_17 + sentence_vecs_18).div(3)'''
    print(sentence_vecs.shape)
    sentence_vecs.to_pickle('data\grimm\master_vecs_grimm_erw.pkl')

def get_classified():
    sentence_vecs = classifier.load_pickle('data\grimm\master_vecs_grimm_erw.pkl') 
    df = classifier.get_all_df(output_src)
    all_vecs =sentence_vecs.columns
    X=sentence_vecs[all_vecs].values
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    #normalized_X = normalize(scaled_X, norm='l1', axis=1, copy=True)
    print("your vectors are: \n",sentence_vecs.head())
    classifier.classify_this(scaled_X,classifier_nb, df['year'])#for nb restructure vectors for negative values
    #classifier.classify_this(sentence_vecs,classifier_nb)#for nb restructure vectors for negative values

get_classified()
#tokens_to_vec()
