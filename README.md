HLA-CNN and HLA-Vec
=========================
__Author__: ysvang@uci.edu

__Usage:__ python main.py config.ini 

__Overview:__ HLA-CNN tool can be used to make binding prediction on HLA Class I peptides
based on convolutional neural networks and a distributed representation of amino acids, HLA-Vec. 
At a high level, the tool consists of (a) an unsupervised, distributed vector representation learner for
raw peptide sequence, (b) a training mode to learn weights to the classifier, (c) an evaluation mode 
to calculate Spearman's rank correlation coefficient (SRCC) and are under the receiver operating characteristic
curve (AUC), (d) an inference mode to make prediction new peptides.

__Pipeline__: The pipeline is specified in the config.ini file. A config file is required to specify the 
parameters used in the various learning algorithm as well as files and directories.
- _HLA-Vec_: learns an unsupervised, distributed vector representation based on raw peptide sequence
- _train_: The classifier is trained from a set of labeled data (reference Supplementary Information from doi:10.1038/srep32115)
- _evaluate_: Using a labeled test set, the trained models are evaluated in terms of SRCC and AUC.
- _inference_: Given a set of new peptides, predictions are inferred and scores are written out to a result file.

__Notes:__
- Test files are obtained from and currently in the format given by IEDB. http://tools.iedb.org/auto_bench/mhci/weekly/
- Although all columns in the test files are not required, the minimum ones required by the code are Allele, Measurement_type, Peptide_seq,
and Measurement_value (not required if performing inference mode).
 
__License:__ This project is licensed under the MIT License - see the LICENSE.md file for details.
 
__Requirements__:
- Python 2.7.13
- numpy 1.11.3
- pandas 0.19.2
- scipy 0.18.1
- scikit_learn 0.18.1
- gensim 2.3.0
- keras 2.0.6
- theano 0.9.0

__Reference__:
Vang, Y. S. and Xie, X. (2017) HLA class I binding prediction via convolutional neural networks. https://doi.org/10.1093/bioinformatics/btx264
