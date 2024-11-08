import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_mean, scatter_add, scatter_softmax, scatter_max #change for Hyper
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_geometric.utils.num_nodes import maybe_num_nodes
import utils.graph_utils as graph_utils
from collections import deque
from tqdm import tqdm

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)

EPS = 1e-15

class HyperS2V_DQN(nn.Module): 
    def __init__(self, reg_hidden, embed_dim, node_dim, edge_dim, T, w_scale, avg=False):
        '''w_scale=0.01, node_dim=2, edge_dim=4'''
        super(HyperS2V_DQN, self).__init__()
        self.T = T 
        self.embed_dim = embed_dim 
        self.reg_hidden = reg_hidden
        self.avg = avg
    
        self.w_n2l = torch.nn.Parameter(torch.Tensor(node_dim, embed_dim))
        torch.nn.init.normal_(self.w_n2l, mean=0, std=w_scale)

        self.w_e2l = torch.nn.Parameter(torch.Tensor(edge_dim, embed_dim))
        torch.nn.init.normal_(self.w_e2l, mean=0, std=w_scale)
        
        self.p_node_conv = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.p_node_conv, mean=0, std=w_scale)

        self.trans_node_1 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.trans_node_1, mean=0, std=w_scale)

        self.trans_node_2 = torch.nn.Parameter(torch.Tensor(1, embed_dim))
        torch.nn.init.normal_(self.trans_node_2, mean=0, std=w_scale)
        

        if self.reg_hidden > 0:
            self.h1_weight = torch.nn.Parameter(torch.Tensor(2 * embed_dim, reg_hidden))
            torch.nn.init.normal_(self.h1_weight, mean=0, std=w_scale)
            self.h2_weight = torch.nn.Parameter(torch.Tensor(reg_hidden, 1))
            torch.nn.init.normal_(self.h2_weight, mean=0, std=w_scale)
            self.last_w = self.h2_weight
        else:
            self.h1_weight = torch.nn.Parameter(torch.Tensor(2 * embed_dim, 1))
            torch.nn.init.normal_(self.h1_weight, mean=0, std=w_scale)
            self.last_w = self.h1_weight

        self.scatter_aggr = (scatter_mean if self.avg else scatter_add)
        

    def forward(self, data):
        vertex = data.vertex
        edges = data.edges
        state = data.x[:, -1]
        data.x = torch.matmul(data.x, self.w_n2l) 
        data.x = F.relu(data.x)
        edge_attr = torch.matmul(data.edge_weight, self.w_e2l)  
        state = state.unsqueeze(1)
        state_attr = torch.matmul(state, self.trans_node_2)
    
        for _ in tqdm(range(self.T)):
            X = data.x
            N = X.shape[0]
            
            Xve = X[vertex] 
            Xe = scatter(Xve, edges, dim=0, reduce='sum') #part 1 first level sum
        
            Xev = Xe[edges] 
            Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) #part 1 second level sum
            part1 = torch.matmul(Xv, self.p_node_conv)
                        
            edge_rep = F.relu(edge_attr) #part 2
            
            part2 = self.scatter_aggr(edge_rep, vertex, dim=0, dim_size=N) #part 2 aggregation
            
            part2 = torch.matmul(part2, self.trans_node_1)
            
            part1_2 =  torch.add(part1, part2)
           
            data.x = torch.add(part1_2, state_attr) #adding (parts 1 and 2 )and part 3
            
            data.x = F.relu(data.x)
            

        y_potential = self.scatter_aggr(data.x, data.batch, dim=0)
        if data.y is not None: 
            action_embed = data.x[data.y]
            embed_s_a = torch.cat((action_embed, y_potential), dim=-1) # ConcatCols

            last_output = embed_s_a
            if self.reg_hidden > 0:
                
                hidden = torch.matmul(embed_s_a, self.h1_weight)
                last_output = F.relu(hidden)
            q_pred = torch.matmul(last_output, self.last_w)

            return q_pred

        else: 
            rep_y = y_potential[data.batch]
            embed_s_a_all = torch.cat((data.x, rep_y), dim=-1) # ConcatCols

            last_output = embed_s_a_all
            if self.reg_hidden > 0: 
                hidden = torch.matmul(embed_s_a_all, self.h1_weight)
                last_output = torch.relu(hidden)

            q_on_all = torch.matmul(last_output, self.last_w)

            return q_on_all
