from codecs import utf_8_decode, utf_8_encode
from gensim import similarities
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec


def load_model(path):
    model = Word2Vec.load(path)
    return model
def load_vec(path):
    wv=KeyedVectors.load(path, mmap='r')
    return wv

model = load_model("1600_word2vec.model")
vector = load_vec("1600_word2vec.wordvectors")

#print(vector.most_similar('opfer')) ok
#print(vector.most_similar('walfisch')) naja
#print(vector.most_similar("baum")) ok
print(vector.most_similar("würdig")) 
#noch probleme mit umlauten?