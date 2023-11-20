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
    s = np.argmax(delta[:, -1])
    etats_predits = []
    for i in reversed(range(len(genome_test)-1)):
        etats_predits.append(s)
        s = psi[int(s), i+1]
      
    return np.array(etats_predits[::-1])

def get_and_show_coding(etat_predits,annotation_test):
    codants_predits = etat_predits.copy()
    codants_test = annotation_test.copy()
    codants_predits[codants_predits!=0]=1 
    codants_test[codants_test!=0]=1
    fig, ax = plt.subplots(figsize=(15,2))
    ax.plot(codants_predits[100000:200000], label="prediction", ls="--")
    ax.plot(codants_test[100000:200000], label="annotation", lw=3, color="black", alpha=.4)
    plt.legend(loc="best")
    plt.show()
    return codants_predits, codants_test

#Evaluation des performances

def create_confusion_matrix(codants_predits, codants_tests):
    confusion_matrix = np.zeros((2,2))
    for i in range(len(codants_predits)):
        if codants_predits[i] == 1 and codants_tests[i] == 1:
            confusion_matrix[0][0] += 1
        elif codants_predits[i] == 1 and codants_tests[i] == 0:
            confusion_matrix[0][1] += 1
        elif codants_predits[i] == 0 and codants_tests[i] == 1:
            confusion_matrix[1][0] += 1
        elif codants_predits[i] == 0 and codants_tests[i] == 0:
            confusion_matrix[1][1] += 1
    return confusion_matrix

"""
Il semble y avoir une inversion des Faux Négatifs et des Faux Positifs dans le pdf de l'énoncé. 
En effet, un faux négatif est ajouté à la case [1][0] et est obtenu quand la prédiction (états_predits) = 0 et que l'annotation (annotation_test) = 1.
C'est ce qui est fait dans la fonction create_confusion_matrix ci-dessus, pourtant les résultats de l'énoncé indiquent l'inverse.
"""

"""
En ce qui concerne les performances, le modèle n'est pas utilisable en tant que tel. En effet, le nombre de faux positifs est bien trop élevé.
"""

def create_seq(N,Pi,A,B,states,obs):
    """Retourne une séquence d'observations et d'états générée par le HMM"""
    etats = np.zeros(N)
    etats[0] = np.random.choice(states, p=Pi)
    for i in range(1,N):
        etats[i] = np.random.choice(states, p=A[int(etats[i-1])])

    obsers = []    
    for i in range(N):
        obsers.append(np.random.choice(obs, p=B[int(etats[i])]))

    return etats, np.array(obsers)