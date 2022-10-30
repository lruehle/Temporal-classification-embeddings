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




model_skip_1 = load_model("aligned\skip\\1600_skip_align.model")
model_skip_2 = load_model("aligned\skip\\1700_skip_align.model")
model_skip_3 = load_model("aligned\skip\\1800_skip_align.model")
model_skip_erw1 = load_model("aligned\skip_erw\\1600erw_skip.model")
model_skip_erw2 = load_model("aligned\skip_erw\\1700erw_skip.model")
model_skip_erw3 = load_model("aligned\skip_erw\\1800erw_skip.model")




print("vocab length 1600skip Embeddings",len(model_skip_1.wv))
print("vocab length 1700skip Embeddings",len(model_skip_2.wv))
print("vocab length 1800skip Embeddings",len(model_skip_3.wv))
print("vocab length 1600skip_erw Embeddings",len(model_skip_erw1.wv))
print("vocab length 1700skip_erw Embeddings",len(model_skip_erw2.wv))
print("vocab length 1800skip_erw Embeddings",len(model_skip_erw3.wv))
# print("vocab length 1800skip_grimm Embeddings",len(model_skip_grimm.wv))


print("\ntest embedding worked\n")
print("1600_skip most similar 'opfer'",model_skip_1.wv.most_similar("opfer",topn=4))
print("1700_skip most similar 'opfer'",model_skip_2.wv.most_similar("opfer",topn=4))
print("1800_skip most similar 'opfer'",model_skip_3.wv.most_similar("opfer",topn=4))
print("1600_skip_erw most similar 'baum'",model_skip_erw1.wv.most_similar("baum",topn=4))
print("1700_skip_erw most similar 'baum'",model_skip_erw2.wv.most_similar("baum",topn=4))
print("1800_skip_erw most similar 'baum'",model_skip_erw3.wv.most_similar("baum",topn=4))

print("\ntest embedding alignment worked\n")
print("1600_skip most similar to: 1700_skip 'baum'",model_skip_1.wv.most_similar(model_skip_2.wv["baum"],topn=3))
print("1700_skip most similar to: 1800_skip 'baum'",model_skip_2.wv.most_similar(model_skip_3.wv["baum"],topn=3))
print("1800_skip most similar to: 1600_skip 'baum'",model_skip_3.wv.most_similar(model_skip_1.wv["baum"],topn=3))
print("1600_skip_erw most similar to: 1600_skip 'baum'",model_skip_erw1.wv.most_similar(model_skip_1.wv["baum"],topn=3))
print("1700_skip_erw most similar to: 1700_skip 'baum'",model_skip_erw2.wv.most_similar(model_skip_2.wv["baum"],topn=3))
print("1800_skip_erw most similar to: 1800_skip 'baum'",model_skip_erw3.wv.most_similar(model_skip_3.wv["baum"],topn=3))

print("\ntest embedding on logic\n")
print("1600_skip most similar: pos 'koenig','frau', neg: 'mann': ",model_skip_1.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3),"\n")  
print("1700_skip most similar: pos 'koenig','frau', neg: 'mann': ",model_skip_2.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3),"\n")  
print("1800_skip most similar: pos 'koenig','frau', neg: 'mann': ",model_skip_3.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3),"\n")  
print("1600_skip_erw most similar: pos 'koenig','frau', neg: 'mann': ",model_skip_erw1.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3),"\n")  
print("1700_skip_erw most similar: pos 'koenig','frau', neg: 'mann': ",model_skip_erw2.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3),"\n")  
print("1800_skip_erw most similar: pos 'koenig','frau', neg: 'mann': ",model_skip_erw3.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3),"\n")  
