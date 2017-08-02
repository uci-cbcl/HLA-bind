"""
The main driver to the HLA-Vec and HLA-CNN scripts.

 @author: ysvang@uci.edu
"""


import configparser
import sys
from src import HLA_Vec
from src import utils
from src import HLA_CNN

if __name__ == '__main__':
    """ The main driver of the software. Parses user inputs from config.ini and runs
        the pipeline.
    """
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    if utils.str2bool(config['Pipeline']['HLA_Vec']):
        print("Starting to learn a distributed representation of amino acids...")
        HLA_Vec.run(config['HLA-Vec'], config['FilesDirectories'])

    if utils.str2bool(config['Pipeline']['train']):
        print("Starting the training of HLA-CNN...")
        HLA_CNN.train(config['HLA-CNN'], config['FilesDirectories'])

    if utils.str2bool(config['Pipeline']['evaluate']):
        print("Performing evaluation on the test set...")
        HLA_CNN.evaluate(config['FilesDirectories'])

    if utils.str2bool(config['Pipeline']['inference']):
        print("Performing inference on the test set...")
        HLA_CNN.inference(config['FilesDirectories'])
