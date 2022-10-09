from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_preprocess
import os
from smart_open import open


dn = os.path.abspath('create_embeddings.py')
input_dir_src = os.path.join(os.path.dirname(dn),'corpora\processed')
input_src = os.path.join(input_dir_src,'merged_file.csv')

"""def read_input(input_file):
    with open(input_file,encoding="utf-8") as f:
        for line in f:
            yield simple_preprocess(line,deacc=True,min_len=4,max_len=20) #list of str tokens
"""
#from https://rare-technologies.com/word2vec-tutorial/
class My_Sentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname),encoding="utf-8"):
                yield simple_preprocess(line,deacc=True,min_len=4,max_len=20)

sentences = My_Sentences(input_dir_src)

word2v_model = Word2Vec(vector_size=100,
                     window=5,
                     min_count=3,
                     sg=0) #test difference in skip gram & Cbow (Cbow: faster & good for big datasets, skip gram better for rare words)

print(word2v_model) 
word2v_model.build_vocab(sentences)    #better save data as tokens before?             
print(word2v_model)  
word2v_model.train(sentences,total_examples=word2v_model.corpus_count, 
                epochs=5)
word2v_model.save("1600_word2vec.model")#saving model
word_vectors=word2v_model.wv
word_vectors.save("1600_word2vec.wordvectors") 