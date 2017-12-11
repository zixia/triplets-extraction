from gensim.models.word2vec import Word2Vec
import pandas as pd
import pdb

train_data = pd.read_csv('../train.csv')
test_data = pd.read_csv('../test.csv')

sentences = train_data.sentences.tolist()+test_data.sentences.tolist()
sentences = [s.strip().split() for s in sentences]
model = Word2Vec(sentences, size=300, window=5, min_count=5, workers=4)
model.wv.save_word2vec_format('em.txt',fvocab='vocab.txt',binary=False)
pdb.set_trace()

