#Léa MOVSESSIAN 28624266
#Timothé COLIN 21206121

import numpy as np
import matplotlib.pyplot as plt

#Génération de données jouet & construction d'une solution analytique

def gen_data_lin(a, b, sig, N,Ntest):
    
    y_test = np.zeros(Ntest)
    x_test = np.zeros(Ntest)
    
    for i in range(Ntest):
        
        x_test[i] = np.random.rand()
        
    x_test.sort()
    
    for i in range(Ntest):
        
        y_test[i] = a*x_test[i] + b + np.random.normal(0, sig, 1)
        
    
    y_train = np.zeros(N)
    x_train = np.zeros(N)
    
    for i in range(N):
        
        x_train[i] = np.random.rand()
        
    x_train.sort()
    
    for i in range(N):
        
        y_train[i] = a*x_train[i] + b + np.random.normal(0, sig, 1)
        
    return x_train, y_train, x_test, y_test


# Validation des formules analytiques

def modele_lin_analytique(X_train, y_train):
    a = np.cov(X_train,y_train, bias = True)[0][1]/np.var(X_train)
    b = np.mean(y_train) - a*np.mean(X_train)
    return a,b

def calcul_prediction_lin(X,ahat,bhat):
    yhat = np.zeros(X.shape)
    
    for i in range(X.shape[0]):
        yhat[i] = ahat*X[i] + bhat
    
    return yhat
    
def erreur_mc(y, yhat):
    
    erreur = 0
    for i in range(y.shape[0]):
        erreur += (yhat[i] - y[i])**2
        
    return erreur/y.shape[0]

def dessine_reg_lin(X_train, y_train, X_test, y_test,a,b):
    plt.plot(X_test, y_test, 'r.',alpha=0.2,label="test")
    plt.plot(X_train, y_train, '-',label="train",color = "blue")
    ahat, bhat = modele_lin_analytique(X_train, y_train)
    yhat = calcul_prediction_lin(X_test,ahat,bhat)
    plt.plot(X_test, yhat, '-', label = "prediction", color = "green")
    plt.legend()
    
# Formulation au sens des moindres carrés

def make_mat_lin_biais(X_train):
    
    Xe = np.ones((X_train.shape[0],2))
    
    for i in range(X_train.shape[0]):
        Xe[i][0] = X_train[i]
        
    return Xe


def reglin_matriciel(Xe,y_train):
    A = np.dot(np.transpose(Xe),Xe)
    B = np.dot(np.transpose(Xe),y_train)
    return np.linalg.solve(A, B)

def calcul_prediction_matriciel(Xe,w):
    yhat = np.zeros(Xe.shape[0])
    
    for i in range(Xe.shape[0]):
        yhat[i] = w[0]*Xe[i][0] + w[1]
    
    return yhat

#Donnees polynomiales

def gen_data_poly2(a, b, c, sig, N, Ntest):
        
    y_test = np.zeros(Ntest)
    x_test = np.zeros(Ntest)
    
    for i in range(Ntest):
        
        x_test[i] = np.random.rand()
        
    x_test.sort()
    
    for i in range(Ntest):
        
        y_test[i] = a*x_test[i]**2 + b*x_test[i] + c + np.random.normal(0, sig, 1)
        
    
    y_train = np.zeros(N)
    x_train = np.zeros(N)
    
    for i in range(N):
        
        x_train[i] = np.random.rand()
        
    x_train.sort()
    
    for i in range(N):
        
        y_train[i] = a*x_train[i]**2 + b*x_train[i] + c + np.random.normal(0, sig, 1)
        
    return x_train, y_train, x_test, y_test

def make_mat_poly_biais(Xp):
    
    Xe = np.ones((Xp.shape[0],3))
    
    for i in range(Xp.shape[0]):
        Xe[i][0] = Xp[i]**2
        Xe[i][1] = Xp[i]
        
    return Xe

def calcul_prediction_polynomial(Xe, w):
    yhat = np.zeros(Xe.shape[0])
    
    for i in range(Xe.shape[0]):
        yhat[i] = w[0]*Xe[i][0] + w[1]*Xe[i][1] + w[2]
    
    return yhat
    

def dessine_poly_matriciel(Xp_train,yp_train,Xp_test,yp_test,w):
    plt.plot(Xp_test, yp_test, 'r.',alpha=0.2,label="test")
    plt.plot(Xp_train, yp_train, '-',label="train",color = "blue")
    
    Xe = make_mat_poly_biais(Xp_test)
    yphat = calcul_prediction_polynomial(Xe, w)
    plt.plot(Xp_test, yphat, '-', label = "prediction", color = "green")
    plt.legend()
    
    

    

    