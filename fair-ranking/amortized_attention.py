#import cvxpy as cp
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt

from gurobipy import *

from birkhoff import birkhoff_von_neumann_decomposition

def calc_position_attentions(num_items, click_probability=0.5, position_cutoff=10):
  return np.array([click_probability * (1 - click_probability) ** i if i < position_cutoff else 0 for i in range(num_items)])

def sample_ranking(probabalistic_ranking):
  import pdb; pdb.set_trace()
  decomposition = birkhoff_von_neumann_decomposition(probabalistic_ranking)
  weights, matrices = zip(*decomposition)
  print(weights)
  index = rn.choice(len(decomposition), 1, p=weights)[0]
  return matrices[index]

def calc_dcg(item_relevances, k=10):
  dropoff = np.log2(np.arange(len(item_relevances)) + 2.0)
  rel = 2 ** item_relevances - 1
  return np.sum(rel[:k] * 1.0 / dropoff[:k]).item()

def calc_probabalistic_dcg_matrix(item_relevances, k=10):
  dropoff = np.log2(np.arange(len(item_relevances)) + 2.0)
  rel = 2 ** item_relevances - 1
  return np.array([[rel[i] * 1.0 / dropoff[j] for j in range(len(item_relevances))] for i in range(len(item_relevances))])

def calc_probabalistic_dcg(item_relevances, P, k=10):
  return cp.sum(cp.multiply(calc_probabalistic_dcg_matrix(item_relevances, k=10),
                            P))

def calc_cost_matrix(num_items,
                     accumulated_attention,
                     accumulated_relevance,
                     position_attentions,
                     item_relevances):
  return np.abs(np.array([[accumulated_attention[i] + position_attentions[j] - (accumulated_relevance[i] + item_relevances[i]) for j in range(num_items)] for i in range(num_items)]))

def get_sol_x_by_x(x,n,cont=False):
    f = int
    default = 0
    if cont:
        f = float
        default = 0.0
    def myfunc():
        values = []
        for i in range(n):
            for j in range(n):
                if i==j:
                    values.append(default)
                elif (i,j) in x:
                    if type(x[i,j]) == int or type(x[i,j]) == float:
                        values.append(f(x[i,j]))
                    else:
                        values.append(f(x[i,j].X))
                elif (j,i) in x:
                    if type(x[j,i]) == int or type(x[j,i]) == float:
                        values.append(f(1-x[j,i]))                    
                    else:
                        values.append(f(1-x[j,i].X))                    
                elif i < j: # don't care about this pair
                    values.append(default)
                else:
                    values.append(1-default)
        return np.reshape(values,(n,n))
    return myfunc()

def main():
  unfairness = []
  baseline_unfairness = []
  solutions = []
  rankings = []
  num_rankings = 3
  num_items = 5
  dcg_drop_ratio_max = 0.1
  position_attentions = calc_position_attentions(num_items)
  print(position_attentions)
  accumulated_attention = np.zeros(num_items)
  accumulated_relevance = np.zeros(num_items)
  for ranking_num in range(num_rankings):
    item_relevances = np.array([1.0 for i in range(num_items)])
    original_ranked_relevances = np.sort(item_relevances)
    C = calc_cost_matrix(num_items,
                         accumulated_attention,
                         accumulated_relevance,
                         position_attentions,
                         item_relevances)
    AP = Model('amortized')
    P = {}
    for i in range(num_items):
        for j in range(num_items):
            P[i,j] = AP.addVar(lb=0,vtype="C",ub=1,name="P(%s,%s)"%(i,j)) #continuous
    
    for i in range(num_items):
        AP.addConstr(quicksum(P[i,j] for j in range(num_items)) == 1)
    for i in range(num_items):
        AP.addConstr(quicksum(P[j,i] for j in range(num_items)) == 1)
        
    temp1 = calc_probabalistic_dcg_matrix(item_relevances, k=10)
    AP.addConstr(quicksum(temp1[i,j]*P[i,j] for i in range(num_items) for j in range(num_items)) >= 
                 dcg_drop_ratio_max * calc_dcg(original_ranked_relevances))
    
    AP.update()
    
    #P = cp.Variable((num_items, num_items))
    AP.Params.Method = 3
    #AP.Params.Crossover = 0 
    AP.update()
    AP.setObjective(quicksum(C[i,j]*P[i,j] for i in range(num_items) for j in range(num_items)),GRB.MINIMIZE)
    
    AP.update()
                    
                  
    
    #objective = cp.Minimize(cp.sum(cp.multiply(C, P)))
    #constraints = [calc_probabalistic_dcg(item_relevances, P) >= ,
    #               cp.matmul(np.ones((1, P.shape[0])), P) == np.ones((1, P.shape[1])),
    #               cp.matmul(P, np.ones((P.shape[1], ))) == np.ones((P.shape[0],)),
    #               #0 <= P, P <= 1]
    #prob = cp.Problem(objective, constraints)
    #result = prob.solve(verbose=False, solver=cp.SCS)
    AP.optimize()
    print(C)
    print(P)
    P_value = get_sol_x_by_x(P,num_items,cont=True)
    #import pdb; pdb.set_trace()
    print(f"Iteration {ranking_num}")
    print("P_value")
    print(P_value)
    ranking = sample_ranking(P_value)
    print("ranking")
    print(ranking)
    solutions.append(P_value)
    rankings.append(ranking)
    ranking_indexes = np.where(ranking)[1]
    accumulated_attention += position_attentions[ranking_indexes]
    accumulated_relevance += item_relevances
    unfairness.append(np.sum(np.abs(accumulated_attention - accumulated_relevance)))
    baseline_unfairness.append(np.sum(np.abs(position_attentions - accumulated_relevance)))


if __name__ == "__main__": main()
