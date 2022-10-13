from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_preprocess
import os
from smart_open import open
from gensim.models.callbacks import CallbackAny2Vec
import pandas as pd

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
            #df = pd.read_csv(os.path.join(self.dirname, fname), usecols=['txt']) #for line in df:
                yield simple_preprocess(line,deacc=True,min_len=2,max_len=20)

"""class CSV_Sentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        df = pd.read_csv(os.path.join(input_dir_src, input_src))
        for row in df.iterrows():
            p=row['txt']
            yield simple_preprocess(row['txt'],deacc=True,min_len=2,max_len=20)
sentences = CSV_Sentences(input_dir_src)
# iter over pd.df is not considered good
"""
sentences = My_Sentences(input_dir_src)

  

class LossLogger(CallbackAny2Vec):
    '''Output loss at each epoch'''
    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f'  Loss: {loss}')
        self.epoch += 1
loss_logger = LossLogger()


word2v_model = Word2Vec(vector_size=100,
                     window=5,
                     min_count=3,
                     sg=0) #test difference in skip gram & Cbow (Cbow: faster & good for big datasets, skip gram better for rare words)

print(word2v_model) 
word2v_model.build_vocab(sentences)  #better save data as tokens before?             
print("vocab is done!: ",word2v_model)
word2v_model.train(sentences,
                    total_examples=word2v_model.corpus_count, 
                    callbacks=[loss_logger],
                    compute_loss=True,
                    epochs=10)
word2v_model.save("models\\1600_word2vec.model")#saving model
word_vectors=word2v_model.wv
word_vectors.save("models\\vectors\\1600_word2vec.wordvectors") 