from lib2to3.pgen2.tokenize import tokenize
from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_preprocess
import os
from smart_open import open
from gensim.models.callbacks import CallbackAny2Vec
import pandas as pd

### No.2 in pipeline: train embedding models for each corpus subclass (years)


dn = os.path.abspath('create_embeddings.py')
# input_file_src = os.path.join(os.path.dirname(dn),'corpora\processed\\1600_corpus_proc.csv')
#input_src = os.path.join(input_file_src,'merged_file.csv')


#from https://rare-technologies.com/word2vec-tutorial/
# needs to be iterable & restartable
# simple_preprocess returns tokens

class My_Sentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname),encoding="utf-8"):
            #df = pd.read_csv(os.path.join(self.dirname, fname), usecols=['txt']) #for line in df:
                yield simple_preprocess(line,deacc=True,min_len=3,max_len=20)


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


def create_embedding():
    
    sentences = My_Sentences(os.path.join(os.path.dirname(dn),'corpora\processed'))
    word2v_model = Word2Vec(vector_size=100,
                    window=5,
                    min_count=3,
                    sg=1) 

    print(word2v_model) 
    word2v_model.build_vocab(sentences)            
    print("vocab is done!: ",word2v_model)
    word2v_model.train(sentences,
                        total_examples=word2v_model.corpus_count, 
                        callbacks=[loss_logger],
                        compute_loss=True,
                        epochs=12)
    word2v_model.save("models_skipgram\\1700skip_word2vec.model")#saving model
    #word_vectors=word2v_model.wv
    #word_vectors.save("models_skip_erw\\1700erw_skip_word2vec.wordvectors")
    
# create_embedding("1700")


 