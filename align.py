from codecs import utf_8_decode, utf_8_encode
from pyexpat import model
from gensim import similarities
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from pathlib import Path
import align_embeddings
import os

### No 3 in pipeline -> trigger align_embeddings.py for all models
### align all to the first model
### only the same vocab is being kept


def load_model(path):
    model = Word2Vec.load(path)
    return model
def load_vec(path):
    wv=KeyedVectors.load(path, mmap='r')
    return wv


dn = os.path.abspath('test_vectors.py')
dir_embed= os.path.join(os.path.dirname(dn),'aligned')
base_path = os.path.join(os.path.dirname(dn),'models_century\\1600_word2vec.model')
model_path = os.path.join(os.path.dirname(dn),'models_century')


#automatic try version
for n, file in enumerate(os.listdir(model_path)): #still errors
    #filter out dirs
    if file.endswith(".model"):
        #path= os.path.relpath(os.path.join(dir_embed,file))
        if(file != os.path.basename(base_path)): #"1600_word2vec.model"):
            model_base = load_model(base_path)
            load_path = os.path.join(model_path,file)
            model_to_align = load_model(load_path)
            save_path = os.path.join(dir_embed,Path(os.path.basename(file)).stem+"_aligned.model")
            #align models
            aligned_model= align_embeddings.smart_procrustes_align_gensim(model_base, model_to_align)
            aligned_model.save(save_path)
            base_path = save_path
            print(file+" is done")



#manual version: 
model1 = load_model("models\\1600_word2vec.model")
#model2 = load_model("models\\1700_word2vec.model")
model2 = load_model("aligned\\1700_word2vec.model")
model3 = load_model("aligned\\1800_word2vec.model")
#model3 = load_model("models\\1800_word2vec.model")



#aligning models:
#aligned_model= align_embeddings.smart_procrustes_align_gensim(model2, model3)
#aligned_model.save("aligned\\1800_word2vec.model") 