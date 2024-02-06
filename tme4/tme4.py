import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
from IPython.display import Image

data = pkl.load( open('res/faithful.pkl', 'rb'))

#A

def normale_bidim(x, mu, sig):
    
    x = np.array(x)
        
    return (1/(2*np.pi*np.linalg.det(sig)**(1/2)))*np.exp((-1/2)*np.dot(np.dot((x-mu),np.linalg.inv(sig)),(x-mu).transpose()))

def estimation_nuage_haut_gauche():
    X = data["X"] 
    donnees_x = []
    donnees_y = []
    
    for i in range(X.shape[0]):
        if X[i][0] > 3 and X[i][1] > 65:
            donnees_y.append(X[i][1])
            donnees_x.append(X[i][0])
                
    donnees_y = np.array(donnees_y)
    donnees_x = np.array(donnees_x)
    
    mu = [4.25,80]
    mu = np.array(mu)
    
    return mu, np.cov(donnees_x,donnees_y)

def init(X):
    
    pi = np.array([0.5,0.5])
    moy = np.mean(X, axis = 0)
    mu = np.array([[moy[0]+1, moy[1]+1],[moy[0]-1, moy[1]-1]])
    covar = np.cov(X, rowvar=0)
    sig = np.stack((covar,covar))
    return pi, mu, sig

def Q_i(X, pi, mu, sig):
    Qi = []
    for i in range(pi.size):        
        norm = np.array([normale_bidim(x, mu[i], sig[i]) for x in X])
        Qi.append(pi[i] * norm)
    Qi = np.array(Qi)
    return Qi / Qi.sum(axis = 0)

import numpy as np

import numpy as np

def update_param(X, q, pi, mu, sig):
    pi_u = np.array([q[i].sum() for i in range(pi.size)])
    mu_u = np.array([np.array([(q[i] * X[:,0]).sum() / q[i].sum() , (q[i] * X[:,1]).sum() / q[i].sum()]) for i in range(pi.size)])
    sig_u = np.array([np.matmul((q[i] * (X-mu_u[i]).T) , (X-mu_u[i])) / q[i].sum() for i in range(pi.size)])
    
    return pi_u / pi_u.sum(), mu_u, sig_u

def EM(X, initFunc=init, nIterMax=1000, saveParam=None):
    pi, mu, sig = initFunc(X)
    eps = 1e-3

    i = 0
    nIter = 1000
    
    while i < nIterMax :
        Qi = Q_i(X, pi, mu, sig)
        pi_u, mu_u, sig_u = update_param(X, Qi, pi, mu, sig)
        if saveParam is not None:                                         # détection de la sauvergarde
            if not os.path.exists(saveParam[:saveParam.rfind('/')]):     # création du sous-répertoire
                 os.makedirs(saveParam[:saveParam.rfind('/')])
            pkl.dump({'pi':pi_u, 'mu':mu_u, 'Sig': sig_u},\
                     open(saveParam+str(i)+".pkl",'wb'))                 # sérialisation
        if np.abs(mu_u - mu).sum() <= eps:
            nIter = i
            i = nIterMax
        pi, mu, sig = pi_u, mu_u, sig_u
        i += 1
        
    return nIter, pi_u, mu_u, sig_u

# B

def init_4(X):
    
    pi = np.array([0.25,0.25,0.25,0.25])
    moy = np.mean(X, axis = 0)
    mu = np.array([[moy[0]+1, moy[1]+1],[moy[0]-1, moy[1]+1], [moy[0]+1, moy[1]-1], [moy[0]-1, moy[1]-1]])
    covar = np.cov(X, rowvar=0)
    sig = np.stack((covar,covar,covar,covar))
    return pi, mu, sig

def bad_init_4(X):
    
    pi = np.array([0.25,0.25,0.25,0.25])
    moy = np.mean(X, axis = 0)
    mu = np.array([[moy[0]+4, moy[1]+2],[moy[0]+3, moy[1]+4], [moy[0], moy[1]], [moy[0]-5, moy[1]]])
    covar = np.cov(X, rowvar=0)
    sig = np.stack((covar,covar,covar,covar))
    return pi, mu, sig

# C

def init_B(X):
    pi = np.array([0.1 for i in range(10)])
    theta = np.array([X[i*3:(3*(i+1)),:].mean(axis=0) for i in range(10)])
    return pi, theta

def logpobsBernoulli(Xu,theta):
    theta = np.clip(theta, 1e-5, 1 - 1e-5)
    logp = (Xu * np.log(theta) + (1 - Xu) * np.log(1 - theta)).sum()
    return np.array(logp)

def Q_i_B(X, pi, theta):
    logsumexp = np.array([[logpobsBernoulli(x , theta[i]) + np.log(pi[i]) for x in X] for i in range(theta.shape[0])])
    max = np.max(logsumexp,0)
    q = np.array([[logpobsBernoulli(x , theta[i]) for x in X] for i in range(theta.shape[0])])
    
    return np.exp(q + np.log(pi)[:,None] - max - np.log(np.exp(logsumexp - max).sum(axis=0)))

def update_param_B(X, q, pi, theta):
    pi_u = np.array([q[i].sum() for i in range(pi.size)])
    theta_u = np.array([[(q[i] * X[:,j]).sum()/q[i].sum() for j in range(X.shape[1])] for i in range(pi.size)])
    return pi_u / pi_u.sum(), theta_u

def EM_B(X, initFunc=init_B, nIterMax=100, saveParam=None):
    pi, theta = initFunc(X)
    eps = 1e-3

    i = 0
    nIter = 100

    print("L'algorithme tourne lentement mais il marche, désolé pour l'attente")
    
    while i < nIterMax :
        Qi = Q_i_B(X, pi, theta)
        pi_u, theta_u = update_param_B(X, Qi, pi, theta)
        if saveParam is not None:                                         # détection de la sauvergarde
            if not os.path.exists(saveParam[:saveParam.rfind('/')]):     # création du sous-répertoire
                 os.makedirs(saveParam[:saveParam.rfind('/')])
            pkl.dump({'pi':pi_u, 'mu':mu_u, 'Sig': sig_u},\
                     open(saveParam+str(i)+".pkl",'wb'))                 # sérialisation
        if np.abs(theta-theta_u).max() <= eps:
            nIter = i
            i = nIterMax
        pi, theta = pi_u, theta_u
        i += 1
        if i % 10 == 0 and i != 0:
            print("itération n°" + str(i))
        
    return nIter, pi_u, theta_u

def calcul_purete(X, Y, pi, theta):
    Qi = Q_i_B(X, pi, theta)
    Y_hat = np.argmax(Qi, axis=0)
    clusters = np.unique(Y_hat)

    purete_par_cluster = []
    poids_par_cluster = []

    for cluster in clusters:
        indices_cluster = np.where(Y_hat == cluster)
        Y_cluster = Y[indices_cluster]
        val, count = np.unique(Y_cluster, return_counts=True)

        classe_majoritaire = val[np.argmax(count)]
        purete_cluster = np.max(count) / len(Y_cluster)  
        purete_par_cluster.append(purete_cluster)
        
        poids_cluster = np.sum(count) / X.shape[0]
        poids_par_cluster.append(poids_cluster)

    return np.array(purete_par_cluster), np.array(poids_par_cluster)
