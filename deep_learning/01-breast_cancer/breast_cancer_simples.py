# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:16:57 2023

@author: Dário da Silva Melo
"""
 
import pandas as pd
from sklearn.model_selection import train_test_split

previsores = pd.read_csv('entradas_breast.csv')
classe     = pd.read_csv('saidas_breast.csv')

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25) 


                     