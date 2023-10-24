#Léa MOVSESSIAN 28624266
#Colin TIMOTHE 21206121

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import random

import utils

#Apprentissage d'un modèle CM

def discretise(X, d):
    
    intervalle = 360 / d
    liste = []
    
    for x in X:
        liste.append(np.floor(x/intervalle))
        
    return np.array(liste)

def groupByLabel(Y):
    
    #à modifier avec np.unique et np.where
    
    group = dict()
    for i in range(26):
        group[chr(i + 97)] = []
        
    for i in range(len(Y)):
        group[Y[i]].append(i)

    return group
        
def learnMarkovModel(X, d):
    
    dis = discretise(X, d)
    A = np.zeros((d, d))
    Pi = np.zeros(d)
    
    for i in range(len(dis)):
        for j in range(len(dis[i])):
            if j == 0:
                Pi[int(dis[i][j])] += 1
            else:
                A[int(dis[i][j-1])][int(dis[i][j])] += 1
                   
    A = A / np.maximum(A.sum(1).reshape(d, 1), 1) # normalisation
    Pi = Pi / Pi.sum()
    
    return Pi, A


def learn_all_MarkovModels(X,Y,d):
    
    Pi, A = learnMarkovModel(X, d)
    group = groupByLabel(Y)
    for i in range(26):
        lettre = chr(i+97)
        Pi,A = learnMarkovModel([X[i] for i in group[lettre]],d)
        group[lettre] = (Pi, A)
        
        
    return group

#Stationnarité des CM apprises

def stationary_distribution_freq(Xd,d):
    
    freq = np.zeros(d)
    taille = 0
    
    for i in range(len(Xd)):
        for j in range(len(Xd[i])):
            if j != 0:
                freq[int(Xd[i][j])] += 1
            
        taille += len(Xd[i]) - 1
        
        
    for j in range(len(freq)):
        freq[j] /= taille
        
    return freq

def stationary_distribution_sampling(Pi, A, N):
    
    p = random.random()
    courant = 0
    somme_p = 0
    
    for i in range(len(Pi)):
        somme_p += Pi[i]
        
        if p <= somme_p:
            courant = i
            break
    
    liste = np.zeros(len(Pi))
    
    for i in range(N):
        p = random.random()
        somme_p = 0
        
        for j in range(len(A[courant])):
            
            somme_p += A[courant][j]
            
            if p <= somme_p:
                liste[j] += 1
                courant = j
                break
    
    liste = [liste[i]/(N-1) for i in range(len(liste))]
    return liste

def stationary_distribution_fixed_point(A, epsilon):
    Pi = np.zeros(len(A))
    Pi[0] = 1

    while np.square(np.subtract(Pi, np.dot(Pi, A))).mean() > epsilon:
        Pi = np.dot(Pi, A)

    return np.dot(Pi, A)

def stationary_distribution_fixed_point_VP(A, epsilon=1e-10):
    
    A = A - np.eye(len(A))

    n = A.shape[0]
    AT = A.T
    v = np.ones(n) / n  # Initialisation avec un vecteur uniforme
    
    while True:
        v_new = np.dot(v, AT)
        v_new /= np.linalg.norm(v_new, ord=1)  # Normalisation L1
        if np.linalg.norm(v_new - v, ord=1) < epsilon:
            return v_new
        v = v_new
    

    
    
            
        