
import math
import random
import numpy as np
import matplotlib.pyplot as plt

#1 - La planche de Galton

def bernoulli(p):
    if random.random() <= p:
        return 1
    else:
        return 0
    
def binomiale(n,p):
    X=0
    for i in range(n):
        if random.random() <= p:
            X=X+1
            
    return X


def galton(l,n,p):
    
    tableau = []
    
    for i in range(l):
        tableau.append(binomiale(n,p))
    return np.array(tableau)

def histo_galton(l,n,p):
    
    valeurs = galton(l,n,p)
    plt.hist (valeurs, np.unique(valeurs))

#2 - Visualisation d'indépendance

def normale (k, sigma):
    
    if k%2 == 0:
        return "Erreur, k pair"
    
    x = np.linspace ( -2 * sigma, 2 * sigma, k )
    
    norm = []
    
    for i in x:    
        
        norm.append((1/(sigma*math.sqrt(2*math.pi))) * math.exp(-((i/sigma)**2)/2))
        
    return np.array(norm)
    
    
def proba_affine(k,slope):
    
    tab = []
    
    if k%2 == 0:
        return "Erreur, k pair"
    
    if slope<0:
        return "Erreur, pente trop faible"
    
    if 1/k + (-(k-1)/2)*slope <0:
        return "Erreur, pente trop élevée"
    
    for i in range(k):
        tab.append(1/k+(i-(k-1)/2)*slope)
    
    return np.array(tab)

#3 - 1 - Indépendance de X et T conditionnellement à (Y,Z)

def Pxy(norm, affi):
    
    n = np.shape(norm)[0]
    m = np.shape(affi)[0]
    mat = [[0 for i in range(n)] for j in range(m)]
    for i in range(n):
        for j in range(m):
            mat[i][j] = norm[i]*affi[j]
            
    return np.array(mat)


def calcYZ(P_XYZT):
    
    mat = [[0 for i in range(2)] for j in range(2)]
    
    for y in range(2):
        for z in range(2):
            somme = 0
            for x in range(2):
                for t in range(2):
                    somme += P_XYZT[x][y][z][t]
            
            mat[y][z] = somme
            
    return np.array(mat)
    
def calcXTcondYZ(P_XYZT):
    P_YZ = calcYZ(P_XYZT)
    mat = [[[[0 for i in range(2)] for j in range(2)] for k in range(2)] for l in range(2)]
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for t in range(2):
                    mat[x][y][z][t]=P_XYZT[x][y][z][t]/P_YZ[y][z]
                    
    return np.array(mat)

def calcX_etTcondYZ(P_XYZT):
    P_YZ = calcYZ(P_XYZT)
    P_XcondYZ = np.zeros((2, 2, 2))
    P_TcondYZ = np.zeros((2, 2, 2))
    
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for t in range(2):
                    P_XcondYZ[x][y][z] += P_XYZT[x][y][z][t]
                    P_TcondYZ[t][y][z] += P_XYZT[x][y][z][t]
    
    for x in range(2):
        for y in range(2):
            for z in range(2):
                P_XcondYZ[x][y][z] /= P_YZ[y][z]
                P_TcondYZ[x][y][z] /= P_YZ[y][z]
                    
    return P_XcondYZ, P_TcondYZ

def testXTindepCondYZ(P_XYZT, epsilon=1e-10):
    P_XcondYZ, P_TcondYZ = calcX_etTcondYZ(P_XYZT)
    P_XTcondYZ = calcXTcondYZ(P_XYZT)
    
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for t in range(2):
                    left_side = P_XcondYZ[x][y][z] * P_TcondYZ[t][y][z]
                    right_side = P_XTcondYZ[x][y][z][t]
                    
                    if abs(left_side - right_side) > epsilon:
                        return False
    
    return True

# 3 - 2 Indépendance de X et Y

def testXindepYZ(P_XYZT, epsilon):
    P_XYZ = np.zeros((2, 2, 2))
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for t in range(2):
                    P_XYZ[x][y][z] += P_XYZT[x][y][z][t]
    P_X = np.zeros(2)
    P_YZ = np.zeros((2,2))
    for x in range(2):
        for y in range(2):
            for z in range(2):
                P_X[x] += P_XYZ[x][y][z]
                P_YZ[y][z] += P_XYZ[x][y][z]
                
    for x in range(2):
        for y in range(2):
            for z in range(2):
                left_side = P_X[x] * P_YZ[y][z]
                right_side = P_XYZ[x][y][z]
                    
                if abs(left_side - right_side) > epsilon:
                    return False
    return True
