'''
This script performs the task of finding a distributed representation 
of amino acid using the continuous skip-gram model with 5 sample negative sampling
'''

from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import pickle

min_count = 2
dims = [15,]
windows = [5,]
allWeights = []

for dim in dims:
  for window in windows:
    print('dim: ' + str(dim) + ', window: ' + str(window))
    df = pd.read_csv("Proteins.txt", delim_whitespace=True, header=0)
    df.columns = ['sequence','HLA','target'] 

    # remove any peptide with  unknown variables
    df = df[df.sequence.str.contains('X') == False]
    df = df[df.sequence.str.contains('B') == False]

    df = df.sample(frac=1)
    
    text = list(df.sequence)
    sentences = []
    for aa in range(len(text)):
      sentences.append(list(text[aa]))
    print(len(sentences))
    model = None
    model = Word2Vec(sentences, min_count=min_count, size=dim, window=window, sg=1, iter = 10, batch_words=100)

    vocab = list(model.vocab.keys())
    print vocab

    print model.syn0

    embeddingWeights = np.empty([len(vocab), dim])

    for i in range(len(vocab)):
      embeddingWeights[i,:] = model[vocab[i]]  

    allWeights.append(embeddingWeights)

 
with open('peptideEmbedding.pickle', 'w') as f:
    pickle.dump(allWeights, f)









