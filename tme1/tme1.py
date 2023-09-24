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
        if x > maxi:
            maxi = x
        elif x < mini:
            mini = x
    
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

