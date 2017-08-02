"""
This script performs the task of finding a distributed representation 
of amino acid using the skip-gram model

 @author: ysvang@uci.edu
"""


from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import os
from src import utils

def run(params, dirnames):
    """ Learns the HLA-Vec distributed representation and save object
        for later use with HLA-CNN.
    """
    min_count = int(params['min_count'])
    dim = int(params['vec_dim'])
    window = int(params['window_size'])

    print('Distributed represntation will be learned based on vector dim: ' + str(dim) + ', context window: ' + str(window) + '.')
    df = pd.read_csv(os.path.join(dirnames['train_set']), delim_whitespace=True, header=0)
    df.columns = ['sequence','HLA','target'] 

    # remove any peptide with  unknown amino acids
    df = df[df.sequence.str.contains('X') == False]
    df = df[df.sequence.str.contains('B') == False]

    # simple shuffling of datapoints
    df = df.sample(frac=1)
    
    # extract peptide sequences into a list for use with Word2Vec
    text = list(df.sequence)
    df = []
    sentences = []
    for aa in range(len(text)):
      sentences.append(list(text[aa]))

    print('There are ' + str(len(sentences)) + ' peptide sequences in the source file...')

    model = Word2Vec(sentences, min_count=min_count, size=dim, window=window, sg=utils.str2bool(params['sg_model']), iter = int(params['iter']), batch_words=100)
    
    if not os.path.exists(dirnames['HLA-Vec_embedding']):
        os.makedirs(dirnames['HLA-Vec_embedding'])
    model.save(os.path.join(dirnames['HLA-Vec_embedding'], 'HLA-Vec_Object'))
