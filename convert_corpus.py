import pandas as pd
import os
from smart_open import open
from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
from HanTa import HanoverTagger as ht
from gensim.utils import simple_preprocess
############## new Version, needs testint! ##############
# rerun for different time folders
# or adjust code to run over directories as well and take name from dir as time-variable
# No.1 in processing pipeline


dn = os.path.abspath('convert_corpus.py')
#txt_src = os.path.join(os.path.dirname(dn),'corpora\dta\Belletristik\\1600')
output_src = os.path.join(os.path.dirname(dn),'corpora\processed') #albertinus_landtstoertzer01_1615.txt')
stop_words = stopwords.words('german')

def remove_stop_words(txt):
    token = txt.split()
    return ' '.join([w for w in token if not w in stop_words])
    
def preproc(txt):
    return simple_preprocess(txt,deacc=True,min_len=3, max_len=20)

def lemma_for_grimm(txt):
    #wordnet = WordNetLemmatizer()
    hannover = ht.HanoverTagger('morphmodel_ger.pgz')
    txt = txt.str.replace(".",".\n")
    all_data =[]
    for sentence in txt:
        words_in_sent = []
        sentence = sentence.split()
        for word in sentence:
            k = hannover.analyze(word,taglevel=1)[0]
            words_in_sent.append(k)
        words_in_sent= ' '.join(words_in_sent)
        all_data.append(words_in_sent)
    return all_data

def remove_umlauts(df):
    df = df.replace("/|,|[^\w\s]","",regex=True)
    df = df.replace('\d+', '',regex=True)
    df = df.replace("ä","ae",regex=True)
    df = df.replace("ö","oe",regex=True)
    df = df.replace("ü","ue",regex=True)
    df = df.replace("Ä","Ae",regex=True)
    df = df.replace("Ö","Oe",regex=True)
    df = df.replace("Ü","Ue",regex=True)
    df = df.replace("ß","ss",regex=True)
    return df

##alternative: write data from files to year_file & then process all at once
## dir_name == year
## saves as output_src + year+'_corpus_proc.csv' extension
def proc_files_in_dir(txt_src,output_src,year):
    with open(os.path.join(output_src,year+'_corpus_proc.csv'), 'w+') as outfile: #why open here and still use to_csv below? Fix in Update #w+ should delete file content
        for file in os.listdir(txt_src):
            if file.endswith(".txt"): 
                file_path = f"{txt_src}\{file}"
                df = pd.read_csv(file_path, sep=".\\n", header=None,names=["txt","year"],engine="python") 
                #print(df.head())
                df= df[df['txt'].str.count(' ') > 2] #drop lines with only one or two words -> references to role/active speaker etc.
                df['txt'] = lemma_for_grimm(df['txt'])
                df['txt'] = df['txt'].apply(remove_stop_words)
                df = remove_umlauts(df)
                df['tokenized'] = df['txt'].map(preproc)
                df['year']= year
                df.to_csv(os.path.join(output_src,year+'_corpus_proc.csv'),mode="a",index=False,header=False,sep=";")
                #print(file, " is done")
                #df.to_csv(outfile,mode="a",index=False,header=False,sep=";")
        

### new
'''parent_dir = os.path.join(os.path.dirname(dn),'corpora\dta\Belletristik') # make args
for dir in os.listdir(parent_dir):
    #if dir/file check or NO FILES ON THIS LEVEL
    child_dir = os.path.join(parent_dir,dir)
    proc_files_in_dir(child_dir, output_src, dir)'''
        


## randomize documents
#ds = pd.read_csv(os.path.join(output_src,'merged_file.csv')) 
#ds = ds.sample(frac=1) #random shuffle
#try out df['removed_stop_word']  = df['x'].apply(stop_word_removal)
#ds.to_csv(os.path.join(output_src,'merged_file2.csv'),index = False)



