#Léa Movsessian 28624266
#Timothée Colin 21206121

# Apprentissage 

import numpy as np
import math
import matplotlib.pyplot as plt
import pickle as pkl

def learnML_parameters ( X_train,Y_train ):
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
            
    return mu,sig
        
    
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
    
    classe = 0
    res = - math.inf
    
    for i in range(10):
        vrai = log_likelihood(X_train_i, mu[i], sig[i], eps)
        if  vrai >= res:
            res = vrai
            classe = i
            
    return classe

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

def classificationRate(Y_train,Y_train_hat):
    
    rate = 0
    n = Y_train.shape[0]
    
    mat = matrice_confusion(Y_train, Y_train_hat)
    
    for i in range(10):
        rate += mat[i][i]
    
    rate /= n
    return rate

def classifTest(X_test,Y_test,mu,sig,eps):
    
    print("1 - Classify all test images ...")    
    Y_hat = classify_all_images(X_test, mu, sig, eps)
    
    rate = classificationRate(Y_test, Y_hat)
    print("2 - Classification rate : " + str(rate))
    
    print("Matrice de confusion :")
    mat = matrice_confusion(Y_test, Y_hat)
    
    plt.figure(figsize=(3,3))
    plt.imshow(mat)
    
    return np.where(Y_test!=Y_hat)


#B

def binarisation(X):
    Xb = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i][j] > 0:
                Xb[i][j] = 1
    return Xb

def learnBernouilli(Xb, Y):
    pass

def logpobsBernoulli(X, theta, epsilon):
    pass

def classifBernoulliTest(Xb, Y, theta):
    pass

