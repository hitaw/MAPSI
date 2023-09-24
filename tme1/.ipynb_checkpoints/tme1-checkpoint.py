#Léa MOVSESSIAN, Timothé COLIN CHÊ-QUANG

import numpy as np 
import matplotlib.pyplot as plt

def analyse_rapide(d):

    mean = np.mean(d)
    std = np.std(d)

    quantile = []

    for i in range(0,10):
        quantile.append(np.quantile(d,i/10))

    print("mean = " + str(mean))
    print("std = " + str(std))
    print("quantile = " + str(quantile))


def discretisation_histogramme(d,n):
    maxi = 0
    mini = 500
    liste = []
    
    for donnee in d:
        maxi = max(donnee, maxi)
        mini = min(donnee, mini)
    bornes = []
    intervalle = (maxi-mini)/n
    
    j = mini
    for i in range(0,n+1):
        bornes.append(j)
        j += intervalle
    bornes.sort()
    print("bornes = " + str(bornes))
    
    effectifs = []
    for i in range(0,n):
        effectifs.append(np.where((d>=bornes[i]) & (d<=bornes[i+1]),1,0).sum())
    print("effectifs =" + str(effectifs))
    
    plt.bar(bornes, effectifs+[0], width = 100, color = 'red')

def discretisation_prix_au_km(d,n):
    liste = []
    maxi = 0
    mini = 500
    
    for donnee in d:
        x = donnee[10]/donnee[13]
        liste.append(x)
        maxi = max(x,maxi)
        mini = min(x,mini)
    
    bornes = []
    intervalle = (maxi-mini)/n
    j = mini
    for i in range(0,n+1):
        bornes.append(j)
        j += intervalle
    bornes.sort()
    print("bornes = " + str(bornes))
    
    effectifs = []
    for i in range(0,n):
        effectifs.append(np.where((liste>=bornes[i]) & (liste <= bornes[i+1]),1,0).sum())
    print("effectifs =" + str(effectifs))
    
    plt.bar(bornes, effectifs+[0], width = 0.02, color = 'red')

def loi_jointe_distance_marque(d,n,dico_marques):

    donnees = d[:,13]
    maxi = 0
    mini = 500

    for x in donnees:
        maxi = max(x, maxi)
        mini = min(x, mini)

    bornes = []
    intervalle = (maxi-mini)/n

    j = mini
    for i in range(0,n+1):
        bornes.append(j)
        j += intervalle
    bornes.sort()

    dist_discr = np.zeros(donnees.shape)

    for i in range(0, np.size(donnees)):
        dist_discr[i] = int((donnees[i] - mini)//intervalle)


    mat = np.zeros((len(bornes)-1,len(dico_marques)))

    for i in range(0,len(bornes)-1):
        for j in range(0,len(dico_marques)):
            mat[i][j] = np.where((dist_discr[:]==i) & (d[:,11]==j) ,1,0).sum()
    mat /= mat.sum()

    fig, ax = plt.subplots(1,1)
    plt.imshow(mat, interpolation='nearest')
    ax.set_xticks(np.arange(len(dico_marques)))
    ax.set_xticklabels(dico_marques.keys(),rotation=90,fontsize=8)
    plt.show()
    
def loi_conditionnelle(jointe_dm):
    pass

def check_conditionnelle(dm):
    pass

def trace_trajectoires(data):
    pass

def calcule_matrice_distance(data,coord):
    pass

def calcule_coord_plus_proche(matrice_dist):
    pass

def test_correlation_distance_prix(data):
    pass

def test_correlation_distance_confort(data):
    pass

