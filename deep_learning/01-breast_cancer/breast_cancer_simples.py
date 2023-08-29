"""
Created on Mon Aug 28 10:16:57 2023

@author: Dário da Silva Melo
"""
 
import pandas as pd

import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1

from tensorflow.keras import backend as k # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1

from sklearn.model_selection import train_test_split


#importando por meio do framework Pandas os dados de entrada e saida da nrede neural. 
previsores = pd.read_csv('entradas_breast.csv')
classe     = pd.read_csv('saidas_breast.csv')


#separando a base de dados para treinamento e teste por meio do framework sklearn
previsores_treinamento,previsores_teste,classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25) 

#Criando a Rede Neural
rede_neural_classificador = Sequential([
               
              tf.keras.layers.Dense(units=16, # número de neurônios na camada oculta é dado pela formula: (n° de neurônios da camada de entrada - n° de neurônios da camada de saida) /2
                                    activation = 'relu',# função de ativação na camada oculta 
                                    kernel_initializer = 'random_uniform',# metodo de inicialização dos pesos
                                    input_dim=30),# número de neurônios na camada de entrada. Será igual ao numero de variaveis preditoras.
              
              
              tf.keras.layers.Dense(units=1,# número de neurônios na camada de saida
                                    activation = 'sigmoid')])#função de ativação na camada de saída.



'''
otimizador = tf.keras.optimizers.Adam(lr = 0.001,
                                      decay = 0.0001,
                                      clipvalue = 0.5) 


#Criando a metodologia para ajuste dos pesos.
rede_neural_classificador.compile(optimizer = otimizador,# algoritmo que será utilizado para ajuste dos pesos
                      loss = 'binary_crossentropy', # metodo utilizado para calcular o erro ou perca (Ex.: RMSE, MSE, entropia, etc)
                      metrics = ['binary_accuracy'])# metricia utlizada para medir a precisão da rede.

'''
rede_neural_classificador.compile(optimizer = 'adam',
                                  loss = 'binary_crossentropy',
                                  metrics = ['binary_accuracy'])


#executa o treinamento. fit() significa ajustar
rede_neural_classificador.fit(previsores_treinamento,
                              classe_treinamento,
                              batch_size = 10,# batch_size define que após calcular o erro de 10 registro é atualizado o peso.
                              epochs = 100)# definição do número de épocas ou rodadas de treinamento.
