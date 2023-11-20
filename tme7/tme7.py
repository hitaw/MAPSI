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
    delta[:,0] = np.log(Pi) + np.log(B[:,genome_test[0]])
    
    #récursion

    log_A = np.log(A)
    log_B = np.log(B)
    for t in range(1, len(genome_test)-1):
        for j in range(len(A)):
            best = delta[:,t-1] + log_A[:,j]
            delta[j][t] = np.max(best) + np.log(B[j][genome_test[t]])
            psi[j][t] = np.argmax(best)

    #terminaison
    k = np.argmax(delta[:, -1])

    # Chemin
    q = []
    for i in reversed(range(len(genome_test)-1)):
        q.append(k)
        k = psi[int(k), i+1]
      
    return np.array(q[::-1])

def get_and_show_coding(etat_predits,annotation_test):
    codants_predits = etat_predits[etat_predits!=0]
    codants_test = [annotation_test[annotation_test!=0]]
    etat_predits[etat_predits!=0]=1 
    annotation_test[annotation_test!=0]=1
    fig, ax = plt.subplots(figsize=(15,2))
    ax.plot(etat_predits[100000:200000], label="prediction", ls="--")
    ax.plot(annotation_test[100000:200000], label="annotation", lw=3, color="black", alpha=.4)
    plt.legend(loc="best")
    plt.show()
    return codants_predits, codants_test

#Evaluation des performances

def create_confusion_matrix(codants_predits, codants_tests):
    pass