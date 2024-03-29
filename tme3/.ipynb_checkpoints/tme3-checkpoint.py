#Léa Movsessian 28624266
#Timothée Colin 21206121

# Apprentissage 

import numpy as np
import math
import matplotlib.pyplot as plt
import pickle as pkl

#fonction avec np.mean et np.std
def learnML_parameters(X_train, Y_train):
    n, d = X_train.shape
    mu = np.zeros((10, d))
    sig = np.zeros((10, d))
    cmpt_cla = np.bincount(Y_train, minlength=10)
    
    for i in range(10):
        mu[i] = np.mean(X_train[Y_train == i], axis=0)
    
    for i in range(10):
        sig[i] = np.std(X_train[Y_train == i], axis=0)
    
    return mu, sig

#fonction sans
"""def learnML_parameters ( X_train,Y_train ):
    mu = np.zeros((10,256))
    cmpt_cla = np.zeros(10)
    sig = np.zeros((10,256))
    
    n,d = X_train.shape
    
    for image in range(n):
        classe = Y_train[image]
        cmpt_cla[classe] += 1
        for pixel in range(d):
            mu[classe][pixel] += X_train[image][pixel]
    
    for i in range(10):
        for j in range(256):
            mu[i][j] /= cmpt_cla[i]
            
    for image in range(n):
        classe = Y_train[image]
        for pixel in range(d):
            sig[classe][pixel] += (mu[classe][pixel] - X_train[image][pixel])**2
            
    for i in range(10):
        for j in range(256):
            sig[i][j] /= cmpt_cla[i]
            sig[i][j] = math.sqrt(sig[i][j])
            
    return mu,sig"""
        
    
def log_likelihood(X_train_i, mu_i, sig_i, defeps):
    
    res = 0
    
    for i in range(X_train_i.shape[0]):
        if sig_i[i] != 0:
            res += np.log(2*np.pi*sig_i[i]**2) + (X_train_i[i] - mu_i[i])**2/sig_i[i]**2
        else:
            if defeps == -1:
                res += -2
            else:
                res += np.log(2*np.pi*defeps**2) + (X_train_i[i] - mu_i[i])**2/defeps**2
        
    res *= -1/2
    
    return res

def classify_image(X_train_i, mu, sig, eps):
    res = -np.inf
    likelihoods = np.zeros(10)
    
    for i in range(10):
        likelihoods[i] = log_likelihood(X_train_i, mu[i], sig[i], eps)
    
    return np.argmax(likelihoods)


def classify_all_images(X_train, mu, sig, eps):
    
    classe = np.zeros(X_train.shape[0])
    
    for i in range(X_train.shape[0]):
        classe[i] = classify_image(X_train[i], mu, sig, eps)
        
    return classe

def matrice_confusion(Y_train, Y_train_hat):
    
    mat = np.zeros((10,10))
    
    for i in range(Y_train.shape[0]):
        mat[int(Y_train[i])][int(Y_train_hat[i])] += 1
            
    return mat

def classificationRate(Y_train, Y_train_hat):
    
    mat = matrice_confusion(Y_train, Y_train_hat)
    
    rate = np.trace(mat)/len(Y_train)
    
    return rate


def classifTest(X_test,Y_test,mu,sig,eps):
    
    print("1 - Classify all test images ...")    
    Y_hat = classify_all_images(X_test, mu, sig, eps)
    
    rate = classificationRate(Y_test, Y_hat)
    print("2 - Classification rate : " + str(rate))
    
    print("3 - Matrice de confusion :")
    mat = matrice_confusion(Y_test, Y_hat)
    
    plt.figure(figsize=(3,3))
    plt.imshow(mat)
    
    return np.where(Y_test!=Y_hat)


#B - Modélisation par une loi de Bernoulli

def binarisation(X):
    return np.where(X>0,1,0)

def learnBernoulli(Xb, Y):
    theta = []
    for i in range(10):
        theta.append(Xb[Y==i].mean(axis=0))
    return np.array(theta)

def logpobsBernoulli(X, theta, epsilon):
    theta_sans_0 = np.where(theta < epsilon, epsilon, np.where(theta > (1 - epsilon), 1 - epsilon, theta))
    return np.sum(X * np.log(theta_sans_0) + (1 - X) * np.log(1 - theta_sans_0), axis=1)

def classifBernoulliTest(Xb, Y, theta):
    print("1 - Classify all test images ...") 
    mu,sig = learnML_parameters(Xb,Y,)
    Y_hat = classify_all_images(Xb, mu, sig, -1)
    
    rate = classificationRate(Y, Y_hat)
    print("2 - Classification rate : " + str(rate))
    
    print("3 - Matrice de confusion :")
    mat = matrice_confusion(Y, Y_hat)
    
    plt.figure(figsize=(3,3))
    plt.imshow(mat)

#C - Modélisation des profils de chiffre

def learnGeom(Xg_train, Y, seuil):
    return np.array([1/np.mean(Xg_train[Y==i],axis = 0) for i in range(10)])

def logpobsGeom(Xg_test_i, theta):
    X = theta * (1 - 2 * 1e-4) + 1e-4
    return np.sum(np.log(X) + (Xg_test_i -1) * np.log(1 - X), axis = 1)

def classifyGeom(Xg_train_i, theta):
    return np.argmax(logpobsGeom(Xg_train_i, theta))

