import copy
import time
import random
import math
import statistics
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from multiprocessing import Pool
import halp.undirected_hypergraph as hg

random.seed(123)
np.random.seed(123)


def read_hypergraph(path):
    ''' function to read hypergraph '''
    hypergraph = hg.UndirectedHypergraph()   
    # parents = {}
    # children = {}
    edges = {}
    nodes = set()
    hyperedge_ids = []
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not len(line) or line.startswith('#') or line.startswith('%'):
                continue
            row = line.split(',')
            nodes = []
            for r in range(len(row) - 1):
                nodes.append(int(row[r])) 
            hypergraph.add_nodes(nodes)
            hypergraph.add_hyperedge(nodes  , weight = float(row[-1]))
            
    return hypergraph

def get_hyperedge_weights(hypergraph):
    hyperedge_weights = []
    ''' return a list of hyperedge id and weight '''
    hyperedge_id_list = hypergraph.get_hyperedge_id_set()
    for hid in hyperedge_id_list:
        hyperedge_weights.append(tuple([hid, hypergraph.get_hyperedge_weight(hid)]))
    return hyperedge_weights

def compute_act_prob(graph, v, rumor_act_t_dict, rumor_act_dict, truth_act_t_dict, truth_act_dict):
    rumor_act_prev = rumor_act_dict[v]
    truth_act_prev = truth_act_dict[v]

    x_truth = 1
    x_rumor = 1

    for e in graph.get_star(v):
        p_e = graph.get_hyperedge_weight(e)
        for n in graph.get_hyperedge_nodes(e):
            if str(n) != str(v):
                n_rumor_t = rumor_act_t_dict[n]
                n_truth_t = truth_act_t_dict[n]

                x_rumor = x_rumor * (1 - p_e*n_rumor_t)
                x_truth = x_truth * (1 - p_e*n_truth_t)
                
            
    rumor_inf_prob = (1 - x_rumor)*x_truth #eq 9
    truth_inf_prob = (1 - x_truth)*x_rumor #eq 10
    
    rumor_act_t = (rumor_inf_prob * (1 - truth_inf_prob)) * (1 - rumor_act_prev - truth_act_prev) #eq 3
    truth_act_t = (truth_inf_prob * (1 - rumor_inf_prob)) * (1 - rumor_act_prev - truth_act_prev) #eq 4
    
    rumor_act = rumor_act_prev + rumor_act_t #eq 5
    truth_act = truth_act_prev + truth_act_t #eq 6
    
    return rumor_act_t, rumor_act, truth_act_t, truth_act

def compute_rumor_inf(graph, S, O):
    rumor_inf = 0
    t = 0
    truth_act_dict = {}
    truth_act_t_dict = {}
    rumor_act_dict = {}
    rumor_act_t_dict = {}
    
    for v in graph.get_node_set():
        if v in S:
            truth_act_dict[v] = 1
            truth_act_t_dict[v] = 1
            
            rumor_act_dict[v] = 0
            rumor_act_t_dict[v] = 0
        elif v in O:
            rumor_act_dict[v] = 1
            rumor_act_t_dict[v] = 1
            
            truth_act_dict[v] = 0
            truth_act_t_dict[v] = 0
        else:
            truth_act_dict[v] = 0
            truth_act_t_dict[v] = 0

            rumor_act_dict[v] = 0
            rumor_act_t_dict[v] = 0
    
    A = S + list(O)
    while(t < 2):
        for v in graph.get_node_set():
            rumor_act_t, rumor_act, truth_act_t, truth_act = compute_act_prob(graph, v, rumor_act_t_dict, rumor_act_dict, truth_act_t_dict, truth_act_dict)
            rumor_act_t_dict[v] = rumor_act_t
            rumor_act_dict[v] = rumor_act
            truth_act_t_dict[v] = truth_act_t
            truth_act_dict[v] = truth_act
        t = t + 1
        for v in A:
            for h in graph.get_star(v):
                A = list(set(A + graph.get_hyperedge_nodes(h)))
    for v in A:
        rumor_inf = rumor_inf + rumor_act_dict[v] 
        
    return rumor_inf

def compute_expected_rumor_protection_hyper(graph, S, O):
    rumor_inf_unrestrained = 0
    rumor_inf = 0
    rumor_inf_unrestrained = compute_rumor_inf(graph, [], O)
    rumor_inf = compute_rumor_inf(graph, S, O)
    return rumor_inf_unrestrained - rumor_inf
    
def computeMC_rumor_hyper(graph, S, O, R):
    ''' compute expected rumor spread
        R: number of trials
    '''
    protectors = set(S)
    rumor_originators = O.copy()
    rumor_inf = 0
    rumor_inf_unrestrained = 0
   
    for _ in range(R):
        truth_affected_set = protectors.copy()
        rumor_affected_set = rumor_originators.copy()
        source_set = truth_affected_set.union(rumor_affected_set)
        queue = deque(source_set)
        
        source_set_unrestrained = O.copy()
        queue_unrestrained = deque(source_set_unrestrained)
        while True:
            curr_truth_affected_set = set()
            curr_rumor_affected_set = set()
            curr_source_set = set()
            
            curr_source_set_unrestrained = set()
            
            while len(queue) != 0:
                curr_node = queue.popleft()
                for hid in graph.get_star(curr_node):
                    for n in graph.get_hyperedge_nodes(hid):
                        if not(n in source_set) and random.random() <= graph.get_hyperedge_weight(hid):
                            curr_source_set.add(n)
                            if curr_node in truth_affected_set:
                                curr_truth_affected_set.add(n)
                            else:
                                curr_rumor_affected_set.add(n)
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            truth_affected_set |= curr_truth_affected_set
            rumor_affected_set |= curr_rumor_affected_set
            source_set |= curr_source_set
            
            while len(queue_unrestrained) != 0:
                curr_node_unrestrained = queue_unrestrained.popleft()
                
                for hid1 in graph.get_star(curr_node_unrestrained):
                    for n1 in graph.get_hyperedge_nodes(hid1):
                        if not(n1 in source_set_unrestrained) and random.random() <= graph.get_hyperedge_weight(hid1):
                            curr_source_set_unrestrained.add(n1)
            if len(curr_source_set_unrestrained) == 0:
                break
            queue_unrestrained.extend(curr_source_set_unrestrained)
            source_set_unrestrained |= curr_source_set_unrestrained
        rumor_inf += len(rumor_affected_set)
        rumor_inf_unrestrained += len(source_set_unrestrained)
        
    return (rumor_inf_unrestrained - rumor_inf) / R

def workerMC_rumor_hyper(x):
    return computeMC_rumor_hyper(x[0], x[1], x[2], x[3])