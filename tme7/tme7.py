#Léa MOVSESSIAN 28624266
#Colin TIMOTHE 21206121

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


#Apprentissage de la CMC

def learnHMM(genome_train, annotation_train, nb_etat, nb_observation):
    
    A = np.zeros((nb_etat, nb_etat))
    B = np.zeros((nb_etat, nb_observation))
    
    for i in range(len(annotation_train)-1):
        A[annotation_train[i]][annotation_train[i+1]] += 1
        B[annotation_train[i]][genome_train[i]] +=1
        #Comptage des transitions et des observations
        
    sommeA = 0
    sommeB = 0
    for i in range(nb_etat):
        sommeA = np.sum(A[i])
        sommeB = np.sum(B[i])
        for j in range(nb_etat):
            A[i][j] /= sommeA
        for k in range(nb_observation):
            B[i][k] /= sommeB
        #Normalisation des matrices
            
    return A,B
                
     
#Estimation par Viterbi
        
def viterbi(genome_test,Pi,A,B):
    
    #initialisation
    psi = np.zeros((len(A), len(genome_test)))
    psi[:,0]= -1
    delta = np.zeros((len(A), len(genome_test)))
    for i in range(len(A)):
        delta[i][0] = np.log(Pi[i]) + np.log(B[i][genome_test[i]])
    
    #récursion
    for i in range(len(A)):
        for j in range(1, len(genome_test)-1):
            delta[i][j] = (np.max(delta[:,j-1] + np.log(A[i][j])) + np.log(B[i][genome_test[i]])
            psi[i][j] = np.max(delta[:,j-1] + np.log(A[i][j])

    #terminaison
    S = np.zeros(len(genome_test))                           
    St = np.max(delta[:,len(genome_test)]
    S[len(S)] = St
    for i in range (len(S)):
        S[len(S)-i] = np.max(psi[:,len(S)-1]) * S[len(S)]
    
    etats_predits = S
      
    return etats_predits