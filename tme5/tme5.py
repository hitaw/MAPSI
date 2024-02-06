import utils
import numpy as np
import scipy.stats as stats

#Statistique du chi² conditionnel

def sufficient_statistics(data, dico, X, Y, Z):
    
    table = utils.create_contingency_table(data, dico, X, Y, Z)
    arrays = []
    
    for i in range(table.shape[0]):
        arrays.append(table[i][1])
        
    Nz = [table[i][0] for i in range(table.shape[0])]
    Nxz = [[] for i in range(len(arrays))]
    Nyz = [[0 for i in range(len(arrays[0][0]))] for j in range(len(arrays))]
    for z in range(len(arrays)):
        for x in range(arrays[z].shape[0]):
            Nxz[z].append(np.sum(arrays[z][x]))
            for y in range(len(arrays[z][x])):
                Nyz[z][y] += arrays[z][x][y]

                
    res = 0
        
    for x in range(arrays[0].shape[0]):
        for y in range(arrays[0].shape[1]):
            for z in range(len(arrays)):
                if Nz[z] != 0:
                    n = (Nxz[z][x] * Nyz[z][y])/Nz[z]
                    if n != 0:
                        res += ((arrays[z][x][y] - n)**2)/n
                        
    z_not_0 = 0
    
    for i in range(table.shape[0]):
        if table[i][0] != 0: 
            z_not_0 += 1
                        
    degre = np.abs(len(Nxz[0])-1) * np.abs(len(Nyz[0])-1) * z_not_0
                        
    return (res, degre)

#Test d'indépendance
def indep_score(data, dico, X, Y, Z):
    res, degre = sufficient_statistics(data, dico, X, Y, Z)
    return stats.chi2.sf(res, degre)

#Meilleur candidat pour être parent
def best_candidate(data, dico, X, Z, alpha):
    best_Y = []
    min_p_value = alpha
    
    for Y in range(X):
        p_value = indep_score(data, dico, X, Y, Z)
        if p_value < min_p_value:
            best_Y = [Y]
            min_p_value = p_value
    
    return best_Y

#Création des parents d'un noeud
def create_parents(data, dico, X, alpha):
    Z = []
    while True:
        candidate_list = best_candidate(data, dico, X, Z, alpha)
        if candidate_list:
            Y = candidate_list[0]
            Z.append(Y)
        else:
            break
    return Z

#Apprentissage de la structure d'un réseau bayésien
def learn_BN_structure(data, dico, alpha):
    num_nodes = len(dico)
    BN_structure = []

    for node in range(num_nodes):
        parents = create_parents(data, dico, node, alpha)
        BN_structure.append(parents)

    return BN_structure




    
    



