from gensim import corpora
from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models import KeyedVectors
import gensim.downloader as api
import inspect
import os
from smart_open import open


#get project path
dn = os.path.abspath('fst.py')
print(dn)


#get path to corpus
td = os.path.join(os.path.dirname(dn),'corpora\Laudatio\grimm_aschenputtel_119-126.txt')
corpora_dir = os.path.join(os.path.dirname(dn),'corpora\Laudatio')
temp_dir=os.path.join(os.path.dirname(dn),'tmp')
print(corpora_dir)
print(td)
print(temp_dir)
#tokens = [simple_preprocess(sentence, deacc=True) for sentence in open(td)]
#gensim_dic = corpora.Dictionary()
#gc = [gensim_dic.doc2bow(token, allow_update=True) for token in tokens]
#word_freq = [[(gensim_dic[id], frequence) for id, frequence in couple] for couple in gc]
#print(word_freq)



#multiple doc:
mult_data = os.scandir(corpora_dir)
#corp_txt = [[print(document)]for documents in (open(mult_data, encoding="utf-8"))]
laudatio_files = [os.path.join(corpora_dir,entry.name) for entry in mult_data if entry.is_file()]
mult_data.close()
#print(laudatio_files)



#multiple documents:
class MyCorpora:
    def __iter__(self):
        for entry in laudatio_files:
            for line in open(entry, encoding="utf-8"):
                yield simple_preprocess(line,deacc=True,min_len=4,max_len=20) 

nc = MyCorpora()
mult_dict = corpora.Dictionary(nc)
#print(len(mult_dict))
#for vector in nc:
 #   print(vector)

# further processing
#min_len/stop words macht vielleicht keinen Sinn, da selbst diese Worte Einfluss haben sollen? evtl. überprüfen, wie sich diese Verändern
stoplist = set('in für auf der die das mit weil'.split()) #existieren diese Worte überhaupt? oben ist min_len ja bei 4. => nope stop words sind eh schon draußen
stop_ids =[ #map stopwords to an id
    mult_dict.token2id[stopword]
    for stopword in stoplist
    if stopword in mult_dict.token2id
]
once_ids=[tokenid for tokenid, docfreq in mult_dict.dfs.items() if docfreq==1] #<2?


mult_dict.filter_tokens(once_ids + stop_ids) #remove tokens from once (& stop)
mult_dict.compactify()
#print(len(mult_dict))

def trainsave_model():
    model = Word2Vec(sentences=nc, vector_size=100, window=7, min_count=2, epochs=100)
    model.save("frst_word2vec.model")#saving model
    word_vectors=model.wv
    word_vectors.save("frst_word2vec.wordvectors") #saving vector
#trainsave_model()

def load_model():
    wv=KeyedVectors.load("frst_word2vec.wordvectors", mmap='r')
    return wv
    
vector = load_model()
print(vector)
