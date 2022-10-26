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
'''model1 = load_model("models_century\\1600_word2vec.model")
#model2 = load_model("models\\1700_word2vec.model")
model2 = load_model("aligned\century\\1700_word2vec.model")
model3 = load_model("aligned\century\\1800_word2vec.model")
model_grimm = load_model("aligned\grimm\\1800_grimm_word2vec_aligned.model")
model_erw_1 = load_model("aligned\erw\\test_1600_word2vec.model")
model_erw_2 = load_model("aligned\erw\\test_1700_word2vec.model")
model_erw_3 = load_model("aligned\erw\\test_1800_word2vec.model")
model_old_1 = load_model("aligned\erw\\1600erw_old.model")
model_old_2 = load_model("aligned\erw\\1700erw_old.model")
model_old_3 = load_model("aligned\erw\\1800erw_old.model")'''
#model3 = load_model("models\\1800_word2vec.model")
#vector = load_vec("1600_word2vec.wordvectors")


#print(model1.wv.most_similar('opfer')) #ok
#print(model1.wv.most_similar('walfisch')) #naja
#print(model1.wv.most_similar("baum")) #ok
#print(vector.most_similar("wuerdig")) #ok
#print(vector.most_similar("anno")) # sollte noch bereinigt werden

#print(model1.wv.most_similar(positive=['frau', 'koenig'], negative=['mann'],topn=4)) # sollte noch bereinigt werden
# print(model_grimm.wv.most_similar("baum",topn=3))
'''
# print(model3.wv.most_similar(model_grimm.wv["baum"],topn=3))
print("most similar in 1800 for: vector of 1600 for 'baum': ",model3.wv.most_similar(model1.wv["baum"],topn=3))
#print(model3.wv.most_similar("baum",topn=3))
print("1600 most similar: pos 'vater','tochter', neg: 'kind': ",model1.wv.most_similar(positive=['vater', 'tochter'], negative=['kind'],topn=3))
print("1700 most similar: pos 'vater','tochter', neg: 'kind': ",model2.wv.most_similar(positive=['vater', 'tochter'], negative=['kind'],topn=3))
# print(model_grimm.wv.most_similar(positive=['vater', 'tochter'], negative=['kind'],topn=3))  
'''
# print(model_grimm.wv.most_similar("koenig",topn=3))
#print(model1.wv.similarity("sohn","tochter"),"\n")  
'''print("\n calculating similarity across time:\n")
print("most similar in 1800erw for: vector of 1800 for 'baum': ",model_old_3.wv.most_similar(model3.wv["baum"],topn=3),"\n")  
print("most similar in 1700erw for: vector of 1800 for 'baum': ",model_old_2.wv.most_similar(model3.wv["baum"],topn=3),"\n")  
print("most similar in 1600erw for: vector of 1800 for 'baum': ",model_old_1.wv.most_similar(model3.wv["baum"],topn=3),"\n")  
print("most similar in 1800_grimm for: vector of 1800 for 'baum': ",model_grimm.wv.most_similar(model3.wv["baum"],topn=3),"\n")  
print("most similar in 1800_erw for: vector of 1600 for 'frau': ",model_old_3.wv.most_similar(model1.wv["frau"],topn=3),"\n")  
print("most similar in 1700_erw for: vector of 1600 for 'frau': ",model_old_2.wv.most_similar(model1.wv["frau"],topn=3),"\n")  
print("most similar in 1600_erw for: vector of 1600 for 'frau': ",model_old_1.wv.most_similar(model1.wv["frau"],topn=3),"\n")  
print("most similar in 1800_grimm for: vector of 1800 for 'frau': ",model_grimm.wv.most_similar(model1.wv["frau"],topn=3),"\n")  
print("\n calculating similarity within time:\n")
print("1600 most similar 'baum': ",model1.wv.most_similar("baum",topn=3),"\n")  
print("1700 most similar 'baum': ",model2.wv.most_similar("baum",topn=3),"\n")  
print("1800 most similar 'baum': ",model3.wv.most_similar("baum",topn=3),"\n")  
print("1600_erw most similar 'baum': ",model_old_1.wv.most_similar("baum",topn=3),"\n")  
print("1700_erw most similar 'baum': ",model_old_2.wv.most_similar("baum",topn=3),"\n")  
print("1800_erw most similar 'baum': ",model_old_3.wv.most_similar("baum",topn=3),"\n")  
print("1800_grimm most similar 'baum': ",model_grimm.wv.most_similar("baum",topn=3),"\n")  
print("\n calculating concepts:\n")
print("1600 most similar: pos 'koenig','frau', neg: 'mann': ",model1.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3),"\n")  
print("1600_erw most similar: pos 'koenig','frau', neg: 'mann': ",model_old_1.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3),"\n")  
print("1700_erw most similar: pos 'koenig','frau', neg: 'mann': ",model_old_2.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3),"\n")  
print("1800_erw most similar: pos 'koenig','frau', neg: 'mann': ",model_old_3.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3),"\n")  
print("1800_grimm most similar: pos 'koenig','frau', neg: 'mann': ",model_grimm.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3))
'''

'''print("erw_1800 vs _1800 ",model_old_3.wv.most_similar(model3.wv["gott"],topn=3))
print("erw_1800 vs _1700 ",model_old_3.wv.most_similar(model2.wv["gott"],topn=3))
print("erw_1800 vs _1600",model_old_3.wv.most_similar(model1.wv["gott"],topn=3))
print("erw_1600vs _1800 ",model_old_1.wv.most_similar(model3.wv["gott"],topn=3))
print("erw_1600vs _1700 ",model_old_1.wv.most_similar(model2.wv["gott"],topn=3))
print("erw_1600vs _1600",model_old_1.wv.most_similar(model1.wv["gott"],topn=3))
print("erw_1700 vs _1800 ",model_old_2.wv.most_similar(model3.wv["gott"],topn=3))
print("erw_1700 vs _1700 ",model_old_2.wv.most_similar(model2.wv["gott"],topn=3))
print("erw_1700 vs _1600",model_old_2.wv.most_similar(model1.wv["gott"],topn=3))
print("grimm vs 1600 ",model_grimm.wv.most_similar(model1.wv["gott"],topn=3))'''
# length embedding vocabs
'''
print("vocab length 1600 Embeddings",len(model1.wv))
print("vocab length 1700 Embeddings",len(model2.wv))
print("vocab length 1800 Embeddings",len(model3.wv))
print("vocab length 1600_erw Embeddings",len(model_old_1.wv))
print("vocab length 1700_erw Embeddings",len(model_old_2.wv))
print("vocab length 1800_erw Embeddings",len(model_old_3.wv))
print("vocab length 1800_grimm Embeddings",len(model_grimm.wv))'''

'''
print(model3.wv.most_similar(model_erw.wv["baum"],topn=3))
print(model3.wv.most_similar(model_erw_3.wv["baum"],topn=3))
print(model3.wv.most_similar(model_erw_2.wv["baum"],topn=3))
print(model2.wv.most_similar(model_erw_2.wv["baum"],topn=3))'''


model_skip_1 = load_model("models_skipgram\\1600_skipgram.model")
model_skip_2 = load_model("aligned\skip\\1700_skip_align.model")
model_skip_3 = load_model("aligned\skip\\1800_skip_align.model")

print("1600_skip most similar 'opfer'",model_skip_1.wv.most_similar("opfer",topn=4))
print("1700_skip most similar 'opfer'",model_skip_2.wv.most_similar("opfer",topn=4))
print("1800_skip most similar 'opfer'",model_skip_3.wv.most_similar("opfer",topn=4))
'''
print("1600_skip most similar to: 1700_skip 'baum'",model_skip_1.wv.most_similar(model_skip_2.wv["baum"],topn=3))
print("1700_skip most similar to: 1800_skip 'baum'",model_skip_2.wv.most_similar(model_skip_3.wv["baum"],topn=3))
print("1800_skip most similar to: 1600_skip 'baum'",model_skip_3.wv.most_similar(model_skip_1.wv["baum"],topn=3))
print("1600_skip most similar: pos 'koenig','frau', neg: 'mann': ",model_skip_1.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3),"\n")  
print("1700_skip most similar: pos 'koenig','frau', neg: 'mann': ",model_skip_2.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3),"\n")  
print("1800_skip most similar: pos 'koenig','frau', neg: 'mann': ",model_skip_3.wv.most_similar(positive=['koenig', 'frau'], negative=['mann'],topn=3),"\n")  
'''