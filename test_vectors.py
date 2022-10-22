from codecs import utf_8_decode, utf_8_encode
from pyexpat import model
from gensim import similarities
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from pathlib import Path
import align_embeddings
import os

### 


def load_model(path):
    model = Word2Vec.load(path)
    return model
def load_vec(path):
    wv=KeyedVectors.load(path, mmap='r')
    return wv


dn = os.path.abspath('test_vectors.py')
dir_embed= os.path.join(os.path.dirname(dn),'aligned')
base_path = os.path.join(os.path.dirname(dn),'models\\1600_word2vec.model')
model_path = os.path.join(os.path.dirname(dn),'models')



#model1 = load_model("aligned\\1600_word2vec.model")
model1 = load_model("models_century\\1600_word2vec.model")
model_grimm = load_model("aligned\grimm\\1800_word2vec.model")
model_erw = load_model("aligned\erw\\1800erw_word2vec.model")
model_erw_2 = load_model("aligned\erw\\1700erw_word2vec.model")
model_erw_3 = load_model("aligned\erw\\1600erw_word2vec.model")
#model2 = load_model("models\\1700_word2vec.model")
model2 = load_model("aligned\century\\1700_word2vec.model")
model3 = load_model("aligned\century\\1800_word2vec.model")
#model3 = load_model("models\\1800_word2vec.model")
#vector = load_vec("1600_word2vec.wordvectors")


#print(model1.wv.most_similar('opfer')) #ok
#print(model1.wv.most_similar('walfisch')) #naja
#print(model1.wv.most_similar("baum")) #ok
#print(vector.most_similar("wuerdig")) #ok
#print(vector.most_similar("anno")) # sollte noch bereinigt werden
'''
#print(model1.wv.most_similar(positive=['frau', 'koenig'], negative=['mann'],topn=4)) # sollte noch bereinigt werden
print(model_grimm.wv.most_similar("baum",topn=3))
print(model2.wv.most_similar(model1.wv["baum"],topn=3))
print(model3.wv.most_similar(model_grimm.wv["baum"],topn=3))
print(model3.wv.most_similar(model1.wv["baum"],topn=3))
#print(model3.wv.most_similar("baum",topn=3))
print(model1.wv.most_similar(positive=['vater', 'tochter'], negative=['kind'],topn=3))
print(model2.wv.most_similar(positive=['vater', 'tochter'], negative=['kind'],topn=3))
print(model_grimm.wv.most_similar(positive=['vater', 'tochter'], negative=['kind'],topn=3))  
print(model1.wv.most_similar("koenig",topn=3))
print(model_grimm.wv.most_similar("koenig",topn=3))
#print(model1.wv.similarity("sohn","tochter"))    

print(len(model1.wv))
print(len(model2.wv))
print(len(model3.wv))
print(len(model_grimm.wv))
print(len(model_erw.wv))'''

#print(model1.wv)
print(model3.wv.most_similar(model_erw.wv["baum"],topn=3))
print(model3.wv.most_similar(model_erw_3.wv["baum"],topn=3))
print(model3.wv.most_similar(model_erw_2.wv["baum"],topn=3))
print(model2.wv.most_similar(model_erw_2.wv["baum"],topn=3))
