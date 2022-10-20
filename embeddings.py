from lib2to3.pgen2.tokenize import tokenize
from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_preprocess
import os
from smart_open import open
from gensim.models.callbacks import CallbackAny2Vec
import pandas as pd

### No.2 in pipeline: train embedding models for each corpus subclass (years)


dn = os.path.abspath('create_embeddings.py')
input_dir_src = os.path.join(os.path.dirname(dn),'corpora\processed\\1600_corpus_proc.csv')
#input_src = os.path.join(input_dir_src,'merged_file.csv')


#from https://rare-technologies.com/word2vec-tutorial/
# needs to be iterable & restartable
# simple_preprocess returns tokens
'''
old version, includes time and tokens as well, but reads line by line ->better for memory
class My_Sentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname),encoding="utf-8"):
            #df = pd.read_csv(os.path.join(self.dirname, fname), usecols=['txt']) #for line in df:
                yield simple_preprocess(line,deacc=True,min_len=2,max_len=20)
'''

df = pd.read_csv(input_dir_src, sep=";",header=None,names=["txt","year","tokenized"])
#print(df.head())
tokenized = df.txt
#tokenized = tokenized.astype(str)
tokenized = tokenized.fillna('').astype(str).apply(simple_preprocess,deacc=True,min_len=3,max_len=20)
#print(tokenized)
#print(tokenized.loc[0]) 
sentences = tokenized
#sentences = My_Sentences(input_dir_src)


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

### auto train model for each timeframe
'''
foreach file in processed folder:
    run sentences on corpus
    build model for sentences
    save model & vectors
'''

word2v_model = Word2Vec(vector_size=100,
                    window=5,
                    min_count=3,
                    sg=0) #test difference in skip gram & Cbow (Cbow: faster & good for big datasets, skip gram better for rare words)

print(word2v_model) 
word2v_model.build_vocab(sentences)            
print("vocab is done!: ",word2v_model)
word2v_model.train(sentences,
                    total_examples=word2v_model.corpus_count, 
                    callbacks=[loss_logger],
                    compute_loss=True,
                    epochs=12)
word2v_model.save("models\\1600_word2vec.model")#saving model
word_vectors=word2v_model.wv
word_vectors.save("models\\vectors\\1600_word2vec.wordvectors") 