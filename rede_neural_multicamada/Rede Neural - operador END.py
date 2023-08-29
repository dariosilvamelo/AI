# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:51:51 2023

@author: Dário da Silva Melo

Rede neural para o operador END
"""

import numpy as np

entrada = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])

peso = np.array([0.0,0.0])

saida = np.array([0,0,0,1])

taxa_aprendizagem=0.1

def soma(entrada,peso):
    return entrada.dot(peso)

def stepFunction(soma):
    if soma>=1:
        return 1
    else:
        return 0
    
def ajustePeso(peso_atual, dado_entrada, erro):
    return peso_atual + (taxa_aprendizagem * dado_entrada * erro)    
    

def calculaSaida(entrada,saida):
    somatorio_erro=0
    
    for i in range(len(entrada)):
            
        saida_calculada = stepFunction(soma(entrada[i],peso))
        
        erro = abs(saida[i]-saida_calculada)
        
        somatorio_erro +=erro
        
        for j in range(len(peso)):
            peso[j] = ajustePeso( peso[j], entrada[i][j], erro)
            print('Processo de atualização de pesos...: ' + str(peso[j]))
    
    return somatorio_erro

if __name__=='__main__':
    
    rede_com_erro=1

    while rede_com_erro !=0:
       
        rede_com_erro = calculaSaida(entrada, saida)

    print("===============================================================")
    print("Rede Treinada - O melhor conjunto de pesos: ", peso)
    print("===============================================================")
    print("Para os atributos [0,0] a resposta é ",stepFunction(soma(entrada[0],peso)))
    print("Para os atributos [0,1] a resposta é ",stepFunction(soma(entrada[1],peso)))
    print("Para os atributos [1,0] a resposta é ",stepFunction(soma(entrada[2],peso)))
    print("Para os atributos [1,1] a resposta é ",stepFunction(soma(entrada[3],peso)))
    print("===============================================================")
        
      