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



def align_models(model_base, model_to_align,save_path):
    aligned_model= align_embeddings.smart_procrustes_align_gensim(model_base, model_to_align)
    aligned_model.save(save_path)


#automatic try version - testing not completed
'''for n, file in enumerate(os.listdir(model_path)): #still errors
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
            print(file+" is done")'''



#manual version: 
# model_skip_erw1 = load_model("models_skip_erw\\1600erw_skip_word2vec.model")
# model_skip_1 = load_model("aligned\skip\\1600skip_align2.model")
# model_skip_erw2 = load_model("models_skip_erw\\1700erw_skip_word2vec.model")
# model_skip_2 = load_model("aligned\skip\\1700skip_align2.model")
# model_skip_erw3 = load_model("models_skip_erw\\1800erw_skip_word2vec.model")
# model_skip_3 = load_model("aligned\skip\\1800skip_align2.model")


#aligning models:
# print("aligning 1800erw to 1800\nvocab:\n")
# aligned_model= align_embeddings.smart_procrustes_align_gensim(model_skip_2, model_skip_erw2)
# aligned_model.save("aligned\skip_erw\\1700erw_skip.model")
