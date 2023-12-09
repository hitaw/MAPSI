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
    """
    Apprentissage de la regression logistique par montée de gradient
    """
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
    """
    classification multiclasse
    """
    return np.argmax(Y_pred, axis=1)

# Exercice 4 : Normalisation des données X

def normalize(X):
    """
    normalisation des données
    """    
    return X-1

# Exercice 5 : Régression logistique multi-classe

def pred_lr_multi_class(X, W, B):
    """
    prédiction multiclasse
    """
    
    return np.exp(np.dot(X, W) + B)/np.sum(np.exp(np.dot(X, W) + B), axis=1, keepdims=True)

# 5.2 Entraînement de la régression logistique multi-classe
def to_categorical(Y, K):
    """
    """
    return np.eye(K, dtype=int)[Y]

def rl_gradient_ascent_multi_class(X, Y, eta, numEp, verbose=0):
    """
    Entraînement de la régression logistique par montée de gradient pour la classification multi-classe.
    
    Arguments :
    X : ndarray, shape (N, d), les données d'entraînement
    Y : ndarray, shape (N,), les classes originales
    eta : float, le taux d'apprentissage
    numEp : int, le nombre d'époches (itérations)
    verbose : int, niveau de verbosité (0 : pas de messages, 1 : messages d'information)
    
    Returns :
    W : ndarray, shape (d, K), les poids appris pour chaque classe
    B : ndarray, shape (K,), les biais appris pour chaque classe
    """
    K = len(np.unique(Y))  # Nombre de classes
    N, d = X.shape  # Nombre d'exemples et dimension des données
    
    # Encodage one-hot des classes
    Y_one_hot = to_categorical(Y, K)
    
    # Initialisation des poids et des biais
    W = np.zeros((d, K))
    B = np.zeros(K)
    
    for epoch in range(numEp):
        # Calcul des prédictions
        Y_pred = pred_lr_multi_class(X, W, B)
        
        # Calcul du gradient pour les poids
        dW = (1/N) * np.dot(X.T, (Y_one_hot - Y_pred))
        
        # Calcul du gradient pour les biais
        dB = (1/N) * np.sum(Y_one_hot - Y_pred, axis=0)
        
        # Mise à jour des poids et des biais
        W += eta * dW
        B += eta * dB
        
        if verbose == 1 and (epoch % (numEp/10) == 0 or epoch == numEp-1):
            # Calcul de l'accuracy
            Y_pred_labels = classif_multi_class(Y_pred)
            acc = accuracy(Y_pred_labels, Y)
            print("Epoch", epoch," - Accuracy: {0:.2f} %".format(acc*100))
    
    return W, B

# A corriger
def rl_gradient_ascent_multi_class_batch(X, Y, tbatch, eta, numEp, verbose = 0):
    """
    On va maintenant passer à une descente de gradient stochastique**, qui va calculer le gradient sous un sous-ensemble d'exemples. Par exemple, pour un batch de taille $500$, on va avoir 13 mises à jour par époque au lieu d'une avec la descente standard. Ceci va permettre de faire converger l'algorithme avec moins d'époques.
    """
    K = len(np.unique(Y))  # Nombre de classes
    N, d = X.shape  # Nombre d'exemples et dimension des données
    
    # Encodage one-hot des classes
    Y_one_hot = to_categorical(Y, K)
    
    # Initialisation des poids et des biais
    W = np.zeros((d, K))
    B = np.zeros(K)
    
    for epoch in range(numEp):
        # Mélange des données
        indices = np.arange(N)
        np.random.shuffle(indices)
        X = X[indices]
        Y_one_hot = Y_one_hot[indices]
        
        # Calcul des prédictions
        Y_pred = pred_lr_multi_class(X, W, B)
        
        # Calcul du gradient pour les poids
        dW = (1/tbatch) * np.dot(X[:tbatch].T, (Y_one_hot[:tbatch] - Y_pred[:tbatch]))
        
        # Calcul du gradient pour les biais
        dB = (1/tbatch) * np.sum(Y_one_hot[:tbatch] - Y_pred[:tbatch], axis=0)
        
        # Mise à jour des poids et des biais
        W += eta * dW
        B += eta * dB
        
        if verbose == 1 and (epoch % (numEp/10) == 0 or epoch == numEp-1):
            # Calcul de l'accuracy
            Y_pred_labels = classif_multi_class(Y_pred)
            acc = accuracy(Y_pred_labels, Y)
            print("Epoch", epoch," - Accuracy: {0:.2f} %".format(acc*100))
    
    return W, B


# Exercice 6 : régularisation

# 6.1 Malédiction de la dimensonnalité

def add_random_column(X,d, sig = 1.):
    return np.hstack((X, np.random.randn(len(X),d)*sig))

def dimensionality_curse(X, Y, Xt, Yt):
    """
    Nous vous proposons ici de modifier les données pour ajouter des colonnes de bruit. Montrer que la performances se réduit lorsque l'on augmente le nombre de dimensions fantomes.

    - La fonction d'ajout des données fantomes `add_random_column` est fournie ci-dessous.
    - faites la boucle avec des ajouts de $[0,100,200,400, 1000]$ colonnes et tracer l'évolution des performances en apprentissage et en test.
    - Attention: il faut donc modifier $X$ et $Xt$ avec le même nombre de colonne fantome

    **Note :** C'est dans ce cas de figure -qui correspond à beaucoup d'applications réelles- que la régularisation va aider.
    """
    
    # Ajout de colonnes fantomes
    nb_colonnes = [0,100,200,400,1000]
    accs_train = []
    accs_test = []

    print("Dimensionality curse")
    
    for nb_colonne in nb_colonnes:
        X_train = add_random_column(X, nb_colonne)
        X_test = add_random_column(Xt, nb_colonne)
        
        # Entrainement
        W, B = rl_gradient_ascent_multi_class(X_train, Y, 0.1, 1000)
        
        # Prédiction
        Y_pred_train = pred_lr_multi_class(X_train, W, B)
        Y_pred_test = pred_lr_multi_class(X_test, W, B)
        
        # Calcul de l'accuracy
        Y_pred_train_labels = classif_multi_class(Y_pred_train)
        Y_pred_test_labels = classif_multi_class(Y_pred_test)
        acc_train = accuracy(Y_pred_train_labels, Y)
        acc_test = accuracy(Y_pred_test_labels, Yt)
        
        accs_train.append(acc_train)
        accs_test.append(acc_test)
        
        print("Noise ", nb_colonne, "Accuracy train : {0:.2f} %".format(acc_train*100))
        print("Noise ", nb_colonne, "Accuracy test : {0:.2f} %".format(acc_test*100))
    
    # Affichage
    plt.figure()
    plt.plot(nb_colonnes, accs_train, label="Train")
    plt.plot(nb_colonnes, accs_test, label="Test")
    plt.xlabel("Nombre de colonnes fantomes")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# 6.2 Régularisation, performance & interprétation

def dimensionality_curse_regul(X, Y, Xt, Yt, type, llamdba):
    """
    Lorsque la dimension augmente, nous voyons appraître le phénomène de sur-apprentissage.

    On fait souvent l'hypothèse que ce phénomène est lié à un estimateur trop complexe. Afin de simplifier la fonction de coût, on proposer de régulariser le problème d'apprentissage qui devient:

    $$\arg\max_\theta  \mathcal L - \lambda \Omega(\theta), \qquad \mbox{avec: } \Omega(\theta) = \left\{\begin{array}{cl}
    \sum\limits_{j=1}^d \theta_j^2 & \mbox{ régularisation } L_2 \\
    \sum\limits_{j=1}^d |\theta_j| & \mbox{ régularisation } L_1 \\
    \end{array}
    \right.$$

    La régularisation $L_2$ est plus générale est facile à exploiter. La régularisation $L_1$ permet d'obtenir des solutions parcimonieuses, *i.e.* d'annuler complètement les poids attribués à certaines dimensions d'entrée.
    **Expliquer quelle régularisation est la plus adaptée ici**. \
    $\lambda$ est l'hyper-paramètre qui contrôle le compromis entre simplicité et bonnes predictions. Le gradient du terme de régularisation est le suivant :

    - Régularisation $L_2$  : $\nabla_{\mathbf w} \Omega(\mathbf w) = 2\mathbf w$  
    - Régularisation $L_1$ : $\nabla_{\mathbf w} \Omega(\mathbf w) = sign(\mathbf w)$

    **Mettre en place la régularisation dans la fonction `dimensionality_curse_regul` suivante**
    """

    # Ajout de colonnes fantomes
    nb_colonnes = [0,100,200,400,1000]
    accs_train = []
    accs_test = []

    print("Dimensionality curse regul")
    
    for nb_colonne in nb_colonnes:
        X_train = add_random_column(X, nb_colonne)
        X_test = add_random_column(Xt, nb_colonne)
        
        # Entrainement
        W, B = rl_gradient_ascent_multi_class(X_train, Y, llamdba, 1000)

        if type == "l2":
            W = 2*W
        elif type == "l1":
            W = np.sign(W)
        
        # Prédiction
        Y_pred_train = pred_lr_multi_class(X_train, W, B)
        Y_pred_test = pred_lr_multi_class(X_test, W, B)
        
        # Calcul de l'accuracy
        Y_pred_train_labels = classif_multi_class(Y_pred_train)
        Y_pred_test_labels = classif_multi_class(Y_pred_test)
        acc_train = accuracy(Y_pred_train_labels, Y)
        acc_test = accuracy(Y_pred_test_labels, Yt)
        
        accs_train.append(acc_train)
        accs_test.append(acc_test)
        
        print("Noise ", nb_colonne, "Accuracy train : {0:.2f} %".format(acc_train*100))
        print("Noise ", nb_colonne, "Accuracy test : {0:.2f} %".format(acc_test*100))
    
    # Affichage
    plt.figure()
    plt.plot(nb_colonnes, accs_train, label="Train")
    plt.plot(nb_colonnes, accs_test, label="Test")
    plt.xlabel("Nombre de colonnes fantomes")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()