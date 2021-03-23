# coding: utf-8

from preprocessing import preprocessing
from mida_tf import MidaImputer
import numpy as np
import pandas as pd
import sys


seed = 2021
dataset = 'vowel'
missing_rate = 30
test_size = 0.3
optimizer = 'adam'


def main(seed, dataset, missing_rate, test_size, optimizer):

    data = pd.read_csv('./datasets/{}.csv'.format(dataset), delimiter=',', header=None).values

    x = data[:,:-1]
    y = data[:,-1]

    dim_x = x.shape[1]

    x_trn_missing, x_tst_missing, y_trnval, y_tst = preprocessing(x, y, missing_rate, seed, test_size)

    
    mida = MidaImputer(dim_x, theta=7, dr=0.5)
    mida.fit(x_trn_missing, 500, 32, 'adam')
    x_trn_imputed = mida.transform(x_trn_missing)
    x_tst_imputed = mida.transform(x_tst_missing)
    
    return x_trn_imputed, x_tst_imputed

