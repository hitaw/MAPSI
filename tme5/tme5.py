#Léa MOVSESSIAN 28624266
#Colin TIMOTHE 21206121

import utils
import numpy as np

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