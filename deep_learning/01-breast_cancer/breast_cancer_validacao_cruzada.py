# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 09:17:03 2023

@author: Dário da Silva Melo
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import KFold

X = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')


#Criando a Rede Neural
def criarRede():
    classificador = Sequential([
                   
                              Dense(units              = 16,               # número de neurônios na camada oculta é dado pela formula: (n° de neurônios da camada de entrada - n° de neurônios da camada de saida) /2
                                    activation         = 'relu',           # função de ativação na camada oculta 
                                    use_bias           = True,            # o valor false não vai exitir bias na camada de entrada. Caso o paramentro não seja informado o parametro terá valor True.
                                    kernel_initializer = 'random_uniform', # metodo de inicialização dos pesos
                                    input_dim          = 30),              # número de neurônios na camada de entrada. Será igual ao numero de variaveis preditoras.
                              
                              Dropout(0.1),# zera 20% dos neuronios referente aos dados de entrada.
                              # Dropout é uma técnica de regularização usada principalmente em redes neurais profundas e redes neurais convolucionais (CNNs) para evitar o overfitting (sobreajuste) do modelo aos dados de treinamento. O overfitting ocorre quando um modelo de aprendizado de máquina se ajusta excessivamente aos dados de treinamento e, como resultado, tem dificuldade em generalizar bem para dados não vistos, levando a um desempenho ruim em conjuntos de teste ou dados de validação.
                              # A ideia por trás do dropout é simples: durante o treinamento, aleatoriamente "desligar" (ou "abandonar") uma porcentagem especificada de neurônios em uma camada durante cada passagem de treinamento. Isso significa que, em cada iteração do treinamento, uma parte dos neurônios não contribui para a propagação para a frente ou para a retropropagação do erro. Essa aleatoriedade força a rede a aprender de forma mais robusta, impedindo que qualquer neurônio ou conjunto de neurônios se torne excessivamente dependente de um determinado subconjunto de dados de treinamento.

                              Dense(units              = 16,               # número de neurônios na camada oculta é dado pela formula: (n° de neurônios da camada de entrada - n° de neurônios da camada de saida) /2
                                    activation         = 'relu',           # função de ativação na camada oculta 
                                    use_bias           = True,            # o valor false não vai exitir bias na camada de entrada. Caso o paramentro não seja informado o parametro terá valor True.
                                    kernel_initializer = 'random_uniform'),# metodo de inicialização dos pesos

                              Dropout(0.1),

                              Dense(units              = 1,                # número de neurônios na camada de saida
                                    activation         = 'sigmoid',        #função de ativação na camada de saída.
                                    use_bias           = True)
                         ])


    otimizador = tf.keras.optimizers.Adam(lr        = 0.001,  # taxa de aprendizagem.
                                      clipvalue = 0.5)    #prender o valor. Caso os pesos chegem em valores maiores que 0,5 ou menores que -0,5 a função vai vai colegelar esses valores para não fugir muito do padrão.

    #Criando a metodologia para ajuste dos pesos.
    classificador.compile(
                          optimizer = otimizador,           # algoritmo que será utilizado para ajuste dos pesos
                          loss      = 'binary_crossentropy',# metodo utilizado para calcular o erro ou perca (Ex.: RMSE, MSE, entropia, etc)
                          metrics   = ['binary_accuracy']   # metricia utlizada para medir a precisão da rede.
                         )  
    return classificador
    

# Defina o número de folds para a validação cruzada
num_folds = 10

kf = KFold(n_splits     = num_folds, # O número de subconjuntos para dividir o conjunto de dados.
           shuffle      = True,      # Um booleano que indica se os subconjuntos devem ser embaralhados antes de serem divididos.
           random_state = 42)        # Uma semente aleatória que é usada para gerar os subconjuntos.

# Lista para armazenar as pontuações de validação cruzada
scores = []

# Loop sobre os folds
for train_index, test_index in kf.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
 
    # Crie e compile seu modelo Keras
    model =  criarRede()
    
    # Treine o modelo no conjunto de treinamento
    model.fit(X_train, 
              y_train, 
              epochs     = 100, # definição do número de épocas ou rodadas de treinamento.
              batch_size = 10,  # batch_size define que após calcular o erro de 10 registro é atualizado o peso.
              verbose    = 0)   #O parâmetro verbose para o método fit controla o nível de saída do método. O valor padrão é 0, que significa que nenhuma saída é gerada. Valores maiores geram mais saída.
  
    # Paramentro verbose pode ser:    
    # 0 =	Nenhuma saída
    # 1	=   Resumo do processo de treinamento
    # 2	=   Atualizações mais detalhadas sobre o processo de treinamento
    # 3	=   Atualizações muito detalhadas sobre o processo de treinamento    


    # O model.evaluate() é um método que avalia um modelo de aprendizado de máquina usando um conjunto de dados de teste. Ele recebe dois argumentos:
    # x: Os dados de entrada do conjunto de dados de teste.
    # y: Os dados de saída do conjunto de dados de teste.
    # No código, o _ é usado para ignorar o primeiro valor do dicionário de resultados, que é o custo de teste. A precisão de teste é o segundo valor do dicionário, e é isso que queremos obte
    
    _ , accuracy = model.evaluate(X_test, y_test)
    
    scores.append(accuracy)


# Exiba os resultados
media_precisao = np.mean(scores)
desvio_padrao = np.std(scores)


print("Acurácia média: %.2f%%" % (media_precisao * 100))
print("Desvio padrão: %.2f%%" % (desvio_padrao * 100))

'''
A validação cruzada (cross-validation, em inglês) é uma técnica fundamental 
na área de aprendizado de máquina e ciência de dados. Ela é usada para avaliar 
o desempenho de um modelo de maneira mais robusta, especialmente quando você 
tem um conjunto de dados limitado. A validação cruzada desempenha um papel
importante nas seguintes áreas:

Avaliação de desempenho do modelo: 
A validação cruzada permite estimar o quão bem um modelo se generaliza para
dados não vistos. Ela fornece métricas de desempenho, como acurácia, precisão,
recall, F1-score, etc., que ajudam a determinar o quão bem o modelo funciona 
em situações do mundo real.

Seleção de hiperparâmetros: 
A validação cruzada é usada para otimizar hiperparâmetros do modelo, 
como a taxa de aprendizado, o número de camadas e neurônios em uma rede neural,
o valor de regularização, entre outros. Isso é feito testando diferentes 
combinações de hiperparâmetros em conjuntos de validação e escolhendo 
aqueles que produzem o melhor desempenho médio em diferentes divisões dos dados.

Evitar overfitting: 
A validação cruzada ajuda a detectar se um modelo está superajustando (overfitting)
aos dados de treinamento. Se o desempenho médio em conjuntos de validação 
for significativamente pior do que o desempenho no conjunto de treinamento,
isso pode ser um sinal de overfitting.

Estimativa de incerteza: 
A validação cruzada fornece uma estimativa da incerteza associada às métricas 
de desempenho. Isso ajuda a entender a variabilidade no desempenho do modelo 
devido a diferentes divisões dos dados.

Melhor uso de dados limitados: 
Em conjuntos de dados pequenos, cada exemplo é valioso. A validação cruzada 
permite aproveitar ao máximo os dados disponíveis, dividindo-os em partes 
de treinamento e validação em várias iterações, de modo que todos os exemplos
sejam usados tanto para treinamento quanto para validação.

Existem várias técnicas de validação cruzada, como a validação cruzada k-fold,
a validação cruzada leave-one-out, a validação cruzada estratificada, 
entre outras. A escolha da técnica depende do tamanho do conjunto de dados,
da natureza dos dados e dos objetivos do projeto.

Em resumo, a validação cruzada é uma prática essencial para avaliar, 
ajustar e selecionar modelos de aprendizado de máquina de maneira 
confiável e robusta. Ela ajuda a garantir que os modelos funcionem 
bem em uma variedade de situações e minimiza o risco de superajuste
aos dados de treinamento.
'''