#Léa MOVSESSIAN 28624266
#Colin TIMOTHE 21206121

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
    pi_u = []
    pi_u = np.array([q[i].sum() for i in range(pi.size)])
    mu_u = np.array([np.asarray([(q[i] * X[:,0]).sum() / q[i].sum() , (q[i] * X[:,1]).sum() / q[i].sum()]) for i in range(pi.size)])
    sig_u = np.array([np.matmul((q[i] * (X-mu_u[i]).T) , (X-mu_u[i])) / q[i].sum() for i in range(pi.size)])

    pi_u = pi_u / pi_u.sum()
    
    return pi_u, mu_u, sig_u

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