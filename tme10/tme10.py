#Léa MOVSESSIAN 28624266
#Colin TIMOTHE 21206121

import numpy as np
import copy

def exp(rate):
    result = np.zeros(rate.shape)
    for i in range(len(rate)):
        for j in range(len(rate[0])):
            if rate[i][j] == 0:
                rate[i][j] += 1e-200
            result[i][j] = np.random.exponential(1/rate[i][j],1)
    return result
            
def simulation(graph, sources, maxT):
    
    k = graph[1]
    r = graph[2]
    
    sommets = graph[0].keys()
    ti = [0 for sommet in sommets]
    
    for sommet in sommets:
        if sommet not in sources:
            ti[sommet] = maxT
            
    ti_copie = copy.deepcopy(ti)
    while min(ti_copie) < maxT:
        minimum = ti_copie.index(min(ti_copie))
        
        for sommet in sommets:
            
            if ti_copie[minimum] < ti_copie[sommet]:
                
                arete = (minimum, sommet)

                if arete in k:
                    
                    x = np.random.binomial(1, k[arete])

                    if x == 1:
                        
                        delta = np.random.exponential(r[arete])
                        
                        t = ti_copie[minimum] + delta
                        
                        if t < ti[sommet]:                            
                            ti[sommet] = t
                            ti_copie[sommet] = t
                            
        ti_copie[minimum] = maxT
        
    return ti

def getProbaMC(graph, sources, maxT, nbsimu):
    
    resultat = [0 for sommet in graph[0]]
    
    for i in range(nbsimu):
        ti = simulation(graph, sources, maxT)
        
        for i in range(len(ti)):
            if ti[i] < maxT:
                resultat[i] += 1
                
    return [i/nbsimu for i in resultat]
                
def getPredsSuccs(graph):
    
    preds = dict()
    succs = dict()
    
    for sommet in graph[0]:
        
        preds[sommet] = []
        succs[sommet] = []
        
        for suiv in graph[0]:
            
            succ = (sommet, suiv)
            pred = (suiv, sommet)
            
            if succ in graph[1]:
                
                succs[sommet].append((suiv, graph[1][succ], graph[2][succ]))
            
            if pred in graph[1]:
                
                preds[sommet].append((suiv, graph[1][pred], graph[2][pred]))
                
    return preds, succs


def compute_ab(v, times, preds, maxT, eps=1e-20):
    
    if times[v] == 0:
        return (1,0)
        
    pred_list = preds[v]
    
    alpha = []
    beta = []
    
    for triplet in pred_list:
        if times[v] > times[triplet[0]]:
            beta.append(triplet[1]*np.exp(-triplet[2]*(times[v]-times[triplet[0]]))+1-triplet[1])
        
    beta = np.array(beta)
    log_beta = np.log(beta)
    b = np.sum(log_beta)
        
    if times[v] == maxT:
        a = 1
    else:
        for triplet in pred_list:
            if times[v] > times[triplet[0]]:
                alpha.append(triplet[1]*triplet[2]*np.exp(-triplet[2]*(times[v]-times[triplet[0]])))
           
        alpha = np.array(alpha)
        a = max(eps, np.sum(alpha/beta))
        
    return a, b
            
                         
         
                         
                                                     
                                                                 
                                                                                
            
            
            
            
            
        
        
    
    
               
        
        
        
        
        
        
        
        
    
    
        