from codecs import utf_8_decode, utf_8_encode
from gensim import similarities
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
import align_embeddings

def load_model(path):
    model = Word2Vec.load(path)
    return model
def load_vec(path):
    wv=KeyedVectors.load(path, mmap='r')
    return wv

#model = load_model("aligned\\1700_word2vec.model")
model1 = load_model("models\\1600_word2vec.model")
model2 = load_model("aligned\\1700_word2vec.model")
model3 = load_model("aligned\\1800_word2vec.model")
model4 = load_model("models\\1800_word2vec.model")
#vector = load_vec("1600_word2vec.wordvectors")


#print(vector.most_similar('opfer')) #ok
#print(vector.most_similar('walfisch')) #naja
#print(vector.most_similar("baum")) #ok
#print(vector.most_similar("wuerdig")) #ok
#print(vector.most_similar("anno")) # sollte noch bereinigt werden

#print(vector.most_similar(positive=['grossvater', 'enkel'], negative=['mann'])) # sollte noch bereinigt werden
#print(model1.wv.most_similar(model3.wv["baum"],topn=3))
#print(model1.wv.most_similar(model4.wv["baum"],topn=3))
"""print(model1.wv.most_similar(positive=['vater', 'tochter'], negative=['kind'],topn=3))
print(model2.wv.most_similar(positive=['vater', 'tochter'], negative=['kind'],topn=3))
print(model3.wv.most_similar(positive=['vater', 'tochter'], negative=['kind'],topn=3))  """
print(model1.wv.most_similar("fürnemste",topn=3))
print(model2.wv.most_similar("baum",topn=3))
print(model3.wv.most_similar("gay",topn=3)) 
#print(model1.wv.similarity("sohn","tochter"))    



# test aligning models:
#aligned_model= align_embeddings.smart_procrustes_align_gensim(model1, model2)
#aligned_model.save("aligned\\1800_word2vec.model") #not necessary, model already updated in smart_procrustes function => or is it? scheint doch nötig 
#aligned_model[1].save("aligned\1700_word2vec.model")

#testing:
#print(model1.wv.most_similar(aligned_model.wv["baum"],topn=3))
#print(aligned_model == model2) #true => return value is only model2 aligned to model1