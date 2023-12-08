#Léa MOVSESSIAN 28624266
#Colin TIMOTHE 21206121

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle as pkl


#Exercice 1 : Régression logistique binaire

def labels_tobinary(Y, cl):

    return np.where(Y == cl, 1, 0)

def pred_lr(X, w, b):
    """
    fonction de prédiction pour la régression logistique
    """
    return 1/(1+np.exp(- (np.dot(X, w) + b)))
        

def classify_binary(Y_pred):
    """
    classification
    """
    return np.where(Y_pred > 0.5, 1, 0)


def accuracy(Y_predb, Y_c):
    """
    Fonction pour calculer l'accuracy
    """
    return np.count_nonzero(np.where(Y_c == Y_predb, 1, 0))/Y_c.shape[0]


def rl_gradient_ascent(X, Y, eta, niter_max):
    """
    Apprentissage de la régresion logistique par montée de gradient
    output : w, b, accs, it
    """

    m, n = X.shape
    w = np.zeros(n)
    b = 0
    accs = []
    it = 0

    while it < niter_max:
        pred = pred_lr(X, w, b)
        next_w = w + eta * np.dot(X.T, (Y - pred))
        next_b = b + eta * np.sum(Y - pred)
        accs.append(accuracy(classify_binary(pred), Y))
        if np.max(np.abs(next_w - w)) < eta:
            return next_w, next_b, accs, it
        w = next_w
        b = next_b
        it += 1

    return w, b, accs, it

def visualization(w):
    """
    visualisation du vecteur de poids w
    """
    plt.figure()
    plt.imshow(w.reshape(16,16), interpolation = None, cmap = 'gray')


# Exercice 2 : Passage au multiclasse

def rl_gradient_ascent_one_against_all(X, Y, epsilon, niter_max):
    
    classes = np.unique(Y)
    W = []
    B = []

    for cl in classes:
        Y_cl = labels_tobinary(Y, cl)
        w,b,accs,it = rl_gradient_ascent(X, Y_cl, epsilon, niter_max)
        W.append(w)
        B.append(b)
        pred = pred_lr(X, w, b)
        print("Classe : " + str(cl) + " acc train = " + str(round(accuracy(classify_binary(pred), Y_cl)*100, 2)) + "%")

    return np.array(W).T, np.array(B)

def classif_multi_class(Y_pred):
    pass









