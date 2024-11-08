import numpy as np
import statistics
from multiprocessing import Pool
import random
import time
import utils.graph_utils as graph_utils

random.seed(123)
np.random.seed(123)

class Environment:
    ''' environment that the agents run in '''
    def __init__(self, name, graphs, budget, rumor_originators, method='MC', use_cache=False, training=True):
        self.name = name
        self.graphs = graphs
        self.budget = budget
        self.method = method
        self.rumor_originators = rumor_originators
        
        self.use_cache = use_cache
        if self.use_cache:
            if self.method == 'MC':
                self.influences = {} 
            elif self.method == 'prob':
                self.influences = {}
        self.training = training # whether in training mode or testing mode

    def reset_graphs(self, num_graphs=10):
        raise NotImplementedError()

    def reset(self, idx=None, training=True):
        if idx is None:
            self.graph = random.choice(self.graphs)
        else:
            self.graph = self.graphs[idx]
        self.state = [0 for _ in range(len(self.graph.get_node_set()))]
        node_set = self.graph.get_node_set()
        node_list = list(node_set)
        if training:
            self.rumor_originators = set()
            while len(self.rumor_originators) < self.budget:
                ele = node_list[random.randint(0, len(node_list) - 1)]
                if(ele not in self.rumor_originators):
                    self.rumor_originators.add(ele)

        for ro in self.rumor_originators:
            self.state[ro] = -1
        self.prev_inf = 0 
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.training = training

    def compute_reward(self, S):
        num_process = 5 # number of parallel processes
        num_trial = 10000 # number of trials

        need_compute = True
        if self.use_cache and self.method == 'MC':
            S_str = f"{id(self.graph)}.{','.join(map(str, sorted(S)))}"
            need_compute = S_str not in self.influences
        if need_compute:
            if self.method == 'prob':
                es_inf = graph_utils.compute_expected_rumor_protection_hyper(self.graph, S, self.rumor_originators)
            elif self.method == 'MC':
                with Pool(num_process) as p:
                    es_inf = statistics.mean(p.map(graph_utils.workerMC_rumor_hyper, 
                        [[self.graph, S, self.rumor_originators, int(num_trial / num_process)] for _ in range(num_process)]))
            else:
                raise NotImplementedError(f'{self.method}')

            if self.use_cache and self.method == 'MC':
                self.influences[S_str] = es_inf
        else:
            es_inf = self.influences[S_str]

        reward = es_inf - self.prev_inf
        self.prev_inf = es_inf
        # store reward
        self.rewards.append(reward)

        return reward

    def step(self, node, time_reward=None):
        ''' change state and get reward '''
        # node has already been selected
        if self.state[node] == 1:
            return
        # store state and action
        self.states.append(self.state.copy())
        self.actions.append(node)
        # update state
        self.state[node] = 1
        # calculate reward
        if self.name != 'RC':
            raise NotImplementedError(f'Environment {self.name}')

        S = self.actions
        # whether budget is reached
        done = len(S) >= self.budget

        if self.training:
            reward = self.compute_reward(S)
        else:
            if done:
                if time_reward is not None:
                    start_time = time.time()
                reward = self.compute_reward(S)
                if time_reward is not None:
                    time_reward[0] = time.time() - start_time
            else:
                reward = None

        return (reward, done)
