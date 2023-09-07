# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:21:27 2023

@author: Dário da Silva Melo
"""

import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers       import KerasClassifier


previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede(kernel_initializer, activation, neurons):
    classificador = Sequential([ Dense(units        = neurons,      
                                 activation         = activation,
                                 use_bias           = True,       
                                 kernel_initializer = kernel_initializer, 
                                 input_dim          = 30),  
                          
                                 Dropout(0.2),
                          
                                 Dense(units        = neurons,  
                                 activation         = activation,   
                                 use_bias           = True,   
                                 kernel_initializer = kernel_initializer),

                                 Dropout(0.2),

                                 Dense(units        = 1,
                                 activation         = 'sigmoid',  
                                 use_bias           = True)
                              ])
                         
    return classificador

'''
Quando usamos o modelo Keras através do KerasClassifier, é definindo que os 
argumentos optimizer e loss são hiperparâmetros para otimização de grade usando 
GridSearchCV. Assim os argumentos optimizer e loss devem ser, definidos 
diretamente no modelo KerasClassifier ao configurar a grade de pesquisa.
'''


# o argumento model da classe KerasClassifier
# é uma instância de um modelo Keras que você cria usando o TensorFlow 
# e a API Keras diretamente. Você cria o modelo, adiciona camadas, configura
# hiperparâmetros, compila-o e depois treina o modelo com seus dados.
# Isso é útil quando você deseja total controle sobre a criação e treinamento
# do modelo Keras e não está usando funcionalidades específicas do scikit-learn
# para pesquisa de hiperparâmetros ou validação cruzada.


rede = KerasClassifier(model              = criarRede, 
                       kernel_initializer = 'random_uniform', 
                       activation         = 'relu', 
                       neurons            = 18
                      )


parametros = {'optimizer'          : ['adam', 'sgd'],
              'loss'               : ['binary_crossentropy', 'hinge'],
              'kernel_initializer' : ['random_uniform', 'normal'],
              'activation'         : ['relu', 'tanh'],
              'neurons'            : [16,8],
              'batch_size'         : [10, 30],
              'epochs'             : [50, 100]}



grid_search = GridSearchCV(estimator   = rede,
                           param_grid  = parametros,
                           scoring     = 'accuracy',
                           cv          = 5)


grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_









