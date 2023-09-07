"""
Created on Mon Aug 28 10:16:57 2023

@author: Dário da Silva Melo
"""
 
import pandas as pd

import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics         import confusion_matrix, accuracy_score


#importando por meio do framework Pandas os dados de entrada e saida da nrede neural. 
previsores = pd.read_csv('entradas_breast.csv')
classe     = pd.read_csv('saidas_breast.csv')


#separando a base de dados para treinamento e teste por meio do framework sklearn
previsores_treinamento,previsores_teste,classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25) 

#Criando a Rede Neural
rede_neural_classificador = Sequential([
               
                              Dense(units              = 16,               # número de neurônios na camada oculta é dado pela formula: (n° de neurônios da camada de entrada - n° de neurônios da camada de saida) /2
                                    activation         = 'relu',           # função de ativação na camada oculta 
                                    use_bias           = False,            # o valor false não vai exitir bias na camada de entrada. Caso o paramentro não seja informado o parametro terá valor True.
                                    kernel_initializer = 'random_uniform', # metodo de inicialização dos pesos
                                    input_dim          = 30),              # número de neurônios na camada de entrada. Será igual ao numero de variaveis preditoras.


                              Dense(units              = 16,               # número de neurônios na camada oculta é dado pela formula: (n° de neurônios da camada de entrada - n° de neurônios da camada de saida) /2
                                    activation         = 'relu',           # função de ativação na camada oculta 
                                    use_bias           = False,            # o valor false não vai exitir bias na camada de entrada. Caso o paramentro não seja informado o parametro terá valor True.
                                    kernel_initializer = 'random_uniform'),# metodo de inicialização dos pesos
             

                              Dense(units              = 1,                # número de neurônios na camada de saida
                                    activation         = 'sigmoid',        #função de ativação na camada de saída.
                                    use_bias           = False)
                         ])


otimizador = tf.keras.optimizers.Adam(lr        = 0.001,  # taxa de aprendizagem.
                                      clipvalue = 0.5)    #prender o valor. Caso os pesos chegem em valores maiores que 0,5 ou menores que -0,5 a função vai vai colegelar esses valores para não fugir muito do padrão.

#Criando a metodologia para ajuste dos pesos.
rede_neural_classificador.compile(optimizer = otimizador,           # algoritmo que será utilizado para ajuste dos pesos
                                  loss      = 'binary_crossentropy',# metodo utilizado para calcular o erro ou perca (Ex.: RMSE, MSE, entropia, etc)
                                  metrics   = ['binary_accuracy'])  # metricia utlizada para medir a precisão da rede.
'''

rede_neural_classificador.compile(optimizer = 'adam',# algoritmo que será utilizado para ajuste dos pesos
                                  loss = 'binary_crossentropy',# metodo utilizado para calcular o erro ou perca (Ex.: RMSE, MSE, entropia, etc)
                                  metrics = ['binary_accuracy'])# metricia utlizada para medir a precisão da rede.
'''

#executa o treinamento. fit() significa ajustar
rede_neural_classificador.fit(previsores_treinamento,
                              classe_treinamento,
                              batch_size = 10,  # batch_size define que após calcular o erro de 10 registro é atualizado o peso.
                              epochs     = 100) # definição do número de épocas ou rodadas de treinamento.

predicao = rede_neural_classificador.predict(previsores_teste)

predicao = (predicao>0.5)

precisao = accuracy_score(classe_teste,predicao)

matriz = confusion_matrix(classe_teste, predicao)

resultado = rede_neural_classificador.evaluate(previsores_teste,classe_teste) 

pesos0 = rede_neural_classificador.layers[0].get_weights()
pesos1 = rede_neural_classificador.layers[1].get_weights()
pesos2 = rede_neural_classificador.layers[2].get_weights()

print(pesos0)
print(len(pesos0))