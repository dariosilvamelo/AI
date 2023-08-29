# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 14:53:43 2023

@author: Dário da Silva Melo

"""

import numpy as np
                    # número de registro da base de dados:
                    # [a1,a2] são os atributos preditivos do registro (atributo 1 atributo 2)
entradas = np.array([[0,0],  # Registro 01
                     [0,1],  # Registro 02
                     [1,0],  # Registro 03
                     [1,1]]) # Registro 04



#             Ligação da Camada de Entrada c/ a Camada Oculta
#                         Neurônios da camada Oculta
#                    neurônio1   neurônio2   neurônio3 
pesos0 = np.array([[-0.424,-0.740,-0.961],  # pesos que liga o atributo 1 (a1) a camada oculta
                   [0.358,-0.577,-0.469]]) # pesos que liga o atributo 2 (a2) a camada oculta


#               Ligação da Camada Oculta c/ a Camada de Saída
#                         Neurônios da camada Oculta

#               Neurônio de Saida  
pesos1 = np.array([[-0.017],  # peso que liga o neurônio1 (camada oculta) ao neurônio de saída
                   [-0.893],  # peso que liga o neurônio2 (camada oculta) ao neurônio de saída
                   [0.148]]) # peso que liga o neurônio3 (camada oculta) ao neurônio de saída


'''
# gerador de pesos aleatórios
pesos0 = 2*np.random.random((2,3)) - 1
pesos1 = 2*np.random.random((3,1)) - 1
'''

#             Neurônio de Saida  (saída esperada)
saidas = np.array([[0],
                   [1],
                   [1],
                   [0]])


# os pesos são inicializados aleatoriamente.


taxaAprendizagem = 0.5
momento = 1
epocas  = 1000000




def funcaoAtivacao(somatorio):
    '''funções introduzem componente não linear nas redes neurais,
   fazendo que elas possam aprender mais do que relações lineares
   entre as variáveis dependentes e independentes, tornando-as capazes
   de modelar também relações não lineares.'''
   
   # função Sigmoide
    f = 1/(1+np.exp(-somatorio))
    return f




def calcular_camada(dados, pesos):
    somatorio = dados.dot(pesos)
    return funcaoAtivacao(somatorio)




def derivada(camada):
    return camada * ( 1 - camada )




def dadosEntrada_X_delta(dados,delta):
    transposta_dados = dados.T 
    return transposta_dados.dot(delta)
 

if __name__ == "__main__":

    for j in range(epocas):    
    
        camada_oculta = calcular_camada(entradas, pesos0)
    
        camada_saida  = calcular_camada(camada_oculta, pesos1)

        erro = saidas - camada_saida
    
        print("Treinamento:")    
        print("Erro.......: ",str( np.mean(np.abs(erro)) * 100 )+" %")
        print("Precisão...: ",str( 100 - (np.mean(np.abs(erro)) * 100 ) )+" %" )    


        # O operador * é usado para multiplicação elemento a elemento entre matrizes.
        # O metodo dot() é ulizado Multiplicação de Matrizes e Produto Escalar.
    
        delta_saida = erro * derivada(camada_saida) # o operador *
    
        delta_oculta = derivada (camada_oculta) * delta_saida.dot(pesos1.T) 
    
        # equação para atualização dos pesos (Algoritmo Backpropagation - retropropagação)
        # 1° atualiza os pesos que liga a camada oculta com a camada de saída
        # 2° atualiza os pesos que liga a camada de entrada com a camada oculta
        # peso(𝑛+1) = 𝑝𝑒𝑠𝑜(𝑛) ∗ 𝑚𝑜𝑚𝑒𝑛𝑡𝑜 + ( 𝑒𝑛𝑡𝑟𝑎𝑑𝑎 ∗ 𝑑𝑒𝑙𝑡𝑎 ∗ 𝑡𝑎𝑥𝑎𝑑𝑒𝑎𝑝𝑟𝑒𝑛𝑑𝑖𝑧𝑎𝑔𝑒𝑚 ) 
    
        p=dadosEntrada_X_delta( camada_oculta, delta_saida )
    
        pesos1 = (pesos1 * momento) + (dadosEntrada_X_delta( camada_oculta, delta_saida ) * taxaAprendizagem)
    
        pesos0 = (pesos0 * momento) + (dadosEntrada_X_delta( entradas     , delta_oculta ) * taxaAprendizagem)
    

    print("=========================================================================")
    print("Rede neural para o operador XOR - O conjunto de pesos após treinamento: ")
    print("Pesos 0: ")
    print(pesos0)
    print("")
    print("Pesos 1:")
    print(pesos1)
    print("")
    print("=========================================================================")
    print("Para os atributos [0,0] a resposta é ",calcular_camada(calcular_camada(entradas[0], pesos0),pesos1))
    print("Para os atributos [0,1] a resposta é ",calcular_camada(calcular_camada(entradas[1], pesos0),pesos1))
    print("Para os atributos [1,0] a resposta é ",calcular_camada(calcular_camada(entradas[2], pesos0),pesos1))
    print("Para os atributos [1,1] a resposta é ",calcular_camada(calcular_camada(entradas[3], pesos0),pesos1))
    print("=========================================================================")
    

        
    
        
    
    
    
 
    


    



