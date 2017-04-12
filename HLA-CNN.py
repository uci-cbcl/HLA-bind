import numpy as np
import pandas as pd
from collections import OrderedDict
import pickle
import time

from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import stats
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Convolution1D
from keras.utils import np_utils
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping 

#np.random.seed(100)  # for reproducibility

start = time.time()

def build_training_matrix(peptide, peptide_n_mer):

    df = pd.read_csv("Proteins.txt", delim_whitespace=True, header=0)
    df.columns = ['sequence','HLA','target'] 

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


def build_test_matrix(peptide, peptide_n_mer, test_measurement, iedb_ref):
    
    pep_key = peptide + '-' + str(peptide_n_mer)
    
    if test_measurement == 'binary':
        peptide_dict = {'A*02:01-9': ('test_2016-05-03', 'test_2015-08-07_1028928'),
                        'B*15:02-9': 'test_B1502',
                        'B*27:05-9': 'test_B2705',
                        'B*07:02-9': 'test_B0702',
                        'B*27:03-9': 'test_B2703'}
                        
        peptide_file = peptide_dict[pep_key]

        if peptide == 'A*02:01' and iedb_ref == '1029824':
            peptide_file = peptide_file[0]

        if peptide == 'A*02:01' and iedb_ref == '1028928':
            peptide_file = peptide_file[1]
             
        test_df = pd.read_csv(peptide_file, delim_whitespace=True)
        test_peptide = test_df.Peptide_seq
        trueY = test_df['Measurement_value']
        test_df['Measurement_value'] = np.where(test_df.Measurement_value == 1.0, 1, 0) 
    else:
        peptide_dict = {'A*02:01-9': 'test_A0201_9mer',
                        'A*02:01-10': 'test_A0201_10mer',
                        'B*57:01-9': 'test_B5701',
                        'A*02:02-9': 'test_A0202',
                        'A*02:03-9': 'test_A0203_9mer',
                        'A*02:03-10': 'test_A0203_10mer',
                        'A*02:06-9': 'test_A0206_9mer',
                        'A*02:06-10': 'test_A0206_10mer',
                        'A*68:02-9': 'test_A6802_9mer',
                        'A*68:02-10': 'test_A6802_10mer'}
        
        peptide_file = peptide_dict[pep_key]
        test_df = pd.read_csv(peptide_file, delim_whitespace=True)
        test_peptide = test_df.Peptide_seq
        trueY = -test_df['Measurement_value']
        test_df['Measurement_value'] = np.where(test_df.Measurement_value < 500.0, 1, 0) 
        
    test_target = test_df.Measurement_value
    test_target = test_target.as_matrix()
    return test_peptide, test_target, trueY
    

# Build up the amino acid embedding and put into dataframe format
aa_dict = OrderedDict([
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


# function to map each letter in the amino acid sequence to it's integer mapping 
def aa_integerMapping(peptideSeq):
  peptideArray = []
  for aa in peptideSeq:
    peptideArray.append(aa_dict[aa])
  return np.asarray(peptideArray)


''' ******************* user input here **************************** ''' 
peptide = 'A*02:03'
peptide_n_mer = 10
seqMatrix, targetMatrix = build_training_matrix(peptide, peptide_n_mer)

test_measurement = 'ic50'  # can be either 'binary' or 'ic50'
# this variable is only required by A*02:01 allele with 9-mers as
# there are two test dataset with this specification, thus need to specify
# which test
iedb_ref = '1028928'
test_peptide, test_target, trueY = build_test_matrix(peptide, peptide_n_mer, test_measurement, iedb_ref)

''' **************************************************************** '''
 
# map the training peptide sequences to their integer index
featureMatrix = np.empty((0, peptide_n_mer), int)
for num in range(len(seqMatrix)):
  featureMatrix = np.append(featureMatrix, [aa_integerMapping(seqMatrix.iloc[num])], axis=0)

# map the test peptide sequences to their integer index
testMatrix = np.empty((0, peptide_n_mer), int)
for num in range(len(test_peptide)):
  testMatrix = np.append(testMatrix, [aa_integerMapping(test_peptide.iloc[num])], axis=0)

# create training and test datasets
X_train = featureMatrix
Y_train = targetMatrix
X_test = testMatrix
Y_test = test_target

# make five predictions for test set
predMatrix = np.zeros((5, len(Y_test)))
predScores = np.zeros((5, len(Y_test)))
i = 0

# CNN parameters
batch_size = np.ceil(len(X_train)/100.0) # variable batch size depending on number of data points
nb_epoch = 100
nb_filter = 32
filter_length = 7
dropout = .25 

# load in learned distributed representation
with open('peptideEmbedding.pickle') as f:
    pepEmbedding = pickle.load(f)
    
embedded_dim = pepEmbedding[0].shape
print embedded_dim
embedded_dim = embedded_dim[1]
n_aa_symbols = len(aa_dict)
embedding_weights = np.zeros((n_aa_symbols + 1,embedded_dim))
embedding_weights[1:,:] = pepEmbedding[0]

colors = ['cyan', 'indigo', 'red', 'orange', 'blue']

while True:
    model = None
    # CNN model
    model = Sequential()

    model.add(Embedding(input_dim=n_aa_symbols+1, output_dim = embedded_dim, weights=[embedding_weights], input_length=peptide_n_mer, trainable = True))
    
    model.add(Convolution1D(nb_filter, filter_length, border_mode='same', init='glorot_normal'))
    model.add(Activation(LeakyReLU(.3)))
    model.add(Dropout(dropout))
    
    model.add(Convolution1D(nb_filter, filter_length, border_mode='same', init='glorot_normal'))
    model.add(Activation(LeakyReLU(.3)))
    model.add(Dropout(dropout))
    
    model.add(Flatten())
    model.add(Dense(nb_filter*peptide_n_mer)) 
    model.add(Activation('sigmoid'))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
     
    model.compile(loss='binary_crossentropy',
        optimizer=Adam(lr=.004), 
        metrics=['accuracy'])
      
    earlyStopping = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto')
    mod = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[earlyStopping], shuffle=True, validation_data=(X_test, Y_test))
    modLoss = mod.history['loss']
      
    # check to make sure optimization didn't diverged
    if ~np.isnan(modLoss[-1]):
        predScores[i,:] = np.squeeze(model.predict(X_test))
        
        i += 1
        if i >4:
            break


predScoresAvg = np.average(predScores, axis=0)
mean_fpr, mean_tpr, mean_thresholds = roc_curve(Y_test, predScoresAvg, pos_label=1)
mean_auc = auc(mean_fpr, mean_tpr)

# final prediction is based on average score of 5 predictions
rho, pValue = stats.spearmanr(trueY, predScoresAvg)
print('polling avg srcc: ', rho)
print('avg auc: ', mean_auc)

print('The script took {0} second !'.format(time.time() - start))
