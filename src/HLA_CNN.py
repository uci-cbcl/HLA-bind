"""
This file implements the convolutional neural network to train,
evaluate, and make inference prediction.

 @author: ysvang@uci.edu
"""


import numpy as np
import pandas as pd
from collections import OrderedDict
import re
import os
import io

from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import stats

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Conv1D
from keras.utils import np_utils
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.models import load_model

from gensim.models import Word2Vec

# Specify an ordering of amino acids for vectorizing peptide sequence
aa_idx = OrderedDict([
    ('A', 1),
    ('C', 2),
    ('E', 3),
    ('D', 4),
    ('G', 5),
    ('F', 6),
    ('I', 7),
    ('H', 8),
    ('K', 9),
    ('M', 10),
    ('L', 11),
    ('N', 12),
    ('Q', 13),
    ('P', 14),
    ('S', 15),
    ('R', 16),
    ('T', 17),
    ('W', 18),
    ('V', 19),
    ('Y', 20)
])

def build_training_matrix(fname, peptide, peptide_n_mer):
    """ Reads the training data file and returns the sequences of peptides
        and target values
    """
    df = pd.read_csv(fname, delim_whitespace=True, header=0)
    df.columns = ['sequence', 'HLA', 'target']

    # build training matrix
    df = df[df.HLA == peptide]
    df = df[df['sequence'].map(len) == peptide_n_mer]

    # remove any peptide with  unknown variables
    df = df[df.sequence.str.contains('X') == False]
    df = df[df.sequence.str.contains('B') == False]
    # remap target values to 1's and 0's
    df['target'] = np.where(df.target == 1, 1, 0)

    seqMatrix = df.sequence
    targetMatrix = df.target
    targetMatrix = targetMatrix.as_matrix()
    return seqMatrix, targetMatrix


def build_test_matrix(fname):
    """ Reads the test data file and extracts allele subtype,
        peptide length, and measurement type. Returns these information
        along with the peptide sequence and target values.
    """
    test_df = pd.read_csv(fname, delim_whitespace=True)
    peptide = re.search(r'[A-Z]\*\d{2}:\d{2}', test_df['Allele'][0]).group()
    peptide_length = len(test_df['Peptide_seq'][0])
    measurement_type = test_df['Measurement_type'][0]

    if measurement_type.lower() == 'binary':
        test_df['Measurement_value'] = np.where(test_df.Measurement_value == 1.0, 1, 0)
    else:
        test_df['Measurement_value'] = np.where(test_df.Measurement_value < 500.0, 1, 0)
    test_peptide = test_df.Peptide_seq
    test_target = test_df.Measurement_value
    test_target = test_target.as_matrix()
    return test_peptide, test_target, peptide_length, peptide


def aa_integerMapping(peptideSeq):
    """ maps amino acid to its numerical index
    """
    peptideArray = []
    for aa in peptideSeq:
        peptideArray.append(aa_idx[aa])
    return np.asarray(peptideArray)


def read_in_datasets(dirnames):
    """ Reads the specified train and test files and return the
        relevant design and target matrix for the learning pipeline.
    """
    test_peptide, test_target, peptide_n_mer, peptide = build_test_matrix(dirnames['test_set'])
    seqMatrix, targetMatrix = build_training_matrix(dirnames['train_set'], peptide, peptide_n_mer)

    # map the training peptide sequences to their integer index
    featureMatrix = np.empty((0, peptide_n_mer), int)
    for num in range(len(seqMatrix)):
        featureMatrix = np.append(featureMatrix, [aa_integerMapping(seqMatrix.iloc[num])], axis=0)

    # map the test peptide sequences to their integer index
    testMatrix = np.empty((0, peptide_n_mer), int)
    for num in range(len(test_peptide)):
        testMatrix = np.append(testMatrix, [aa_integerMapping(test_peptide.iloc[num])], axis=0)

    # create training and test datasets
    datasets={}
    datasets['X_train'] = featureMatrix
    datasets['Y_train'] = targetMatrix
    datasets['X_test'] = testMatrix
    datasets['Y_test'] = test_target
    return datasets, peptide_n_mer


def make_predictions(dirnames, datasets):    
    """ Makes inference prediction
    """
    X_test = datasets['X_test']
    predScores = np.zeros((5, len(X_test)))

    for i in range(5):
        model = load_model(os.path.join(dirnames['HLA-CNN_models'], 'hla_cnn_model_' + str(i) + '.hdf5'))
        predScores[i, :] = np.squeeze(model.predict(X_test))

    predScoresAvg = np.average(predScores, axis=0)
    return predScoresAvg


def write_predictions(dirnames, Y_pred):
    """ write out predictions scores and labels to a new file
    """
    testset = pd.read_csv(dirnames['test_set'], delim_whitespace=True, header=0)
    if not os.path.exists(os.path.join(dirnames['results'])):
        os.makedirs(os.path.join(dirnames['results']))
        
    predicted_labels = ["binding" if score >= .5 else "non-binding" for score in Y_pred]
    
    # write prediction to new file
    with open(os.path.join(dirnames['results'], "predictions.csv"), 'w') as f:
        f.write('Peptide_seq' + ',' + 'Predicted_Scores' + ',' + 'Predicted_Labels' + '\n')
        for i in range(len(Y_pred)):
            f.write(str(testset['Peptide_seq'].iloc[i]) + ',' + str(Y_pred[i]) + ',' + predicted_labels[i] + '\n')
    f.close()


def train(params, dirnames):
    """ Trains the HLA-CNN model
    """
    datasets, peptide_n_mer = read_in_datasets(dirnames)

    # CNN parameters
    batch_size = int(np.ceil(len(datasets['X_train']) / 100.0))  # variable batch size depending on number of data points
    epochs = int(params['epochs'])
    nb_filter = int(params['filter_size'])
    filter_length = int(params['filter_length'])
    dropout = float(params['dropout'])
    lr = float(params['lr'])

    # load in learned distributed representation HLA-Vec
    hla_vec_obj = Word2Vec.load(os.path.join(dirnames['HLA-Vec_embedding'], 'HLA-Vec_Object'))
    hla_vec_embd = hla_vec_obj.wv
    embd_shape = hla_vec_embd.syn0.shape
    embedding_weights = np.random.rand(embd_shape[0] + 1, embd_shape[1])
    for key in aa_idx.keys():
       embedding_weights[aa_idx[key],:] = hla_vec_embd[key]
    embedded_dim = embd_shape[1]

    i = 0
    while True:
        model = None
        # CNN model
        model = Sequential()

        model.add(Embedding(input_dim=len(aa_idx) + 1, output_dim=embedded_dim, weights=[embedding_weights], input_length=peptide_n_mer, trainable=True))

        model.add(Conv1D(nb_filter, filter_length, padding='same', kernel_initializer='glorot_normal'))
        model.add(LeakyReLU(.3))
        model.add(Dropout(dropout))

        model.add(Conv1D(nb_filter, filter_length, padding='same', kernel_initializer='glorot_normal'))
        model.add(LeakyReLU(.3))
        model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(nb_filter * peptide_n_mer))
        model.add(Activation('sigmoid'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=lr),
                      metrics=['accuracy'])

        earlyStopping = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto')
        mod = model.fit(datasets['X_train'], datasets['Y_train'], batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[earlyStopping], shuffle=True, validation_data=(datasets['X_test'], datasets['Y_test']))
        modLoss = mod.history['loss']

        # check to make sure optimization didn't diverged
        if ~np.isnan(modLoss[-1]):
            if not os.path.exists(dirnames['HLA-CNN_models']):
                os.makedirs(dirnames['HLA-CNN_models'])
            model.save(os.path.join(dirnames['HLA-CNN_models'], 'hla_cnn_model_' + str(i) + '.hdf5'))

            i += 1
            if i > 4:
                break


def evaluate(dirnames):
    """ Evaluates the test file and calculate SRCC and AUC score.
    """
    datasets,_ = read_in_datasets(dirnames)
    Y_test = datasets['Y_test']
    predMatrix = np.zeros((5, len(Y_test)))
    Y_pred = make_predictions(dirnames, datasets)

    mean_fpr, mean_tpr, mean_thresholds = roc_curve(Y_test, Y_pred, pos_label=1)
    mean_auc = auc(mean_fpr, mean_tpr)

    rho, pValue = stats.spearmanr(Y_test, Y_pred)
    print('SRCC: ' + str(round(rho, 3)))
    print('AUC: ' + str(round(mean_auc,3)))


def inference(dirnames):
    """ Makes inference prediction on the test file.
    """
    datasets,_ = read_in_datasets(dirnames)
    Y_pred = make_predictions(dirnames, datasets)
    write_predictions(dirnames, Y_pred)

