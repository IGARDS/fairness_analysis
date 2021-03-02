#import cvxpy as cp
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import pandas as pd
import copy

from gurobipy import *

from birkhoff import birkhoff_von_neumann_decomposition

def calc_position_attentions(num_items, click_probability=0.5, position_cutoff=10):
    return np.array([click_probability * (1 - click_probability) ** i if i < position_cutoff else 0 for i in range(num_items)])

def IDCG(r):
    k = len(r)
    i = np.arange(1,k+1)
    return np.sum((2**r-1)/np.log2(i+1))

def get_sol_x_by_x(x,n,cont=False):
    f = int
    default = 0
    if cont:
        f = float
        default = 0.0
    values = []
    for i in range(n):
        for j in range(n):
            values.append(f(x[i,j].X))
    return np.reshape(values,(n,n))

def constraint_lhs(X,item_relevances,k,num_items):
    #import pdb; pdb.set_trace()
    total = 0
    for i in range(num_items):
        for j in range(k):
            total += (2**item_relevances[i]-1)/np.log2(j+2)*X[i,j]
    return total
    #return np.sum((2**item_relevances[i] - 1)/np.log2(j+2)*X[i,j] for j in range(k) for i in range(num_items))

def solution2ranking(solution):
    ranking = pd.DataFrame(solution).apply(lambda x: np.where(x == 1)[0][0],axis=1)
    return ranking.values
    
import scipy.stats as stats
    
# model 3
def model_3(series_item_relevances,position_attentions,theta=1,k=10,item_names=None):
    num_rankings = len(series_item_relevances)
    series_item_relevances = copy.deepcopy(series_item_relevances)
    # Make sure each sums to 1.0
    for ranking_num in range(num_rankings):
        item_relevances = series_item_relevances[ranking_num]
        series_item_relevances[ranking_num] = series_item_relevances[ranking_num]/np.sum(series_item_relevances[ranking_num])
        
    position_attentions = position_attentions/np.sum(position_attentions)  
        
    num_items = len(series_item_relevances[0])
    
    unfairness = []
    solutions = []
    
    accumulated_attention = np.zeros(num_items) # A
    accumulated_relevance = np.zeros(num_items) # R
    
    for ranking_num in range(num_rankings):
        item_relevances = series_item_relevances[ranking_num]
        order = np.argsort(-1*item_relevances)
        ixs_k = order[:k]
        AP = Model('amortized')
        X = {}
        for i in range(num_items):
            for j in range(num_items):
                X[i,j] = AP.addVar(vtype="BINARY",name="X(%s,%s)"%(i,j)) #binary

        for i in range(num_items):
            AP.addConstr(quicksum(X[i,j] for j in range(num_items)) == 1)
        for i in range(num_items):
            AP.addConstr(quicksum(X[j,i] for j in range(num_items)) == 1)

        AP.addConstr(quicksum((2**item_relevances[i] - 1)/np.log2(j+2)*X[i,j] for i in range(num_items) for j in range(k)) >= 
                     theta * IDCG(item_relevances[ixs_k]))

        AP.update()

        AP.update()
        values = []
        for i in range(num_items):
            values.append([])
            for j in range(num_items):
                values[i].append(accumulated_attention[i] + position_attentions[j] - (accumulated_relevance[i] + item_relevances[i]))
        C = np.abs(np.array(values))
        #C = np.abs(np.array([[accumulated_attention[i] + position_attentions[j] - (accumulated_relevance[i] + item_relevances[i]) for j in range(num_items)] for i in range(num_items)]))
        AP.setObjective(quicksum(C[i,j]*X[i,j] for i in range(num_items) for j in range(num_items)),GRB.MINIMIZE)

        AP.update()
        AP.optimize()
        X_value = get_sol_x_by_x(X,num_items,cont=False)
        solutions.append(X_value)
        ranking = solution2ranking(X_value)
        order2 = np.argsort(ranking)
        tau, p_value = stats.kendalltau(np.argsort(order), np.argsort(order2))
        print(f"{ranking_num}: tau={tau}")
        accumulated_attention += position_attentions[order2]
        accumulated_relevance += item_relevances
        unfairness.append(np.sum(np.abs(accumulated_attention - accumulated_relevance)))
    return solutions, unfairness