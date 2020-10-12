# -*- coding: utf-8 -*-
"""
@author:yym
"""
import numpy as np
import torch

np.random.seed(1)

#this file contains functions to create generate the Reservoir network

def create_random_reservoir(dims, frac_inhibitory, w_matrix, fanout):
    #dims : 3-D dimensions of the reservoir (each of dims[0], dims[1] and dims[2] should be 3 or greater)
    #frac_inhibitory : fraction of inhibitory nodes
    #w_matrix : 2x2 matrix of weights from Excitatory(E) to Inhibitory(I), E to E, I to E and I to I
    #fanout : number of nodes connected to each node
    
    n_nodes = dims[0]*dims[1]*dims[2]
    node_types = np.random.uniform(size=n_nodes)
    node_types = torch.from_numpy(node_types)
    node_types[node_types>frac_inhibitory] = 1
    node_types[node_types<=frac_inhibitory] = -1
    adj_mat = torch.zeros((n_nodes, n_nodes))
    all_connections = []
    all_weights = []
    for i in range(n_nodes):
        z = torch.tensor(i/(dims[0]*dims[1])).int()
        y = torch.tensor((i % (dims[0]*dims[1])/dims[0])).int()
        x = torch.tensor((i % (dims[0]*dims[1])) % dims[0]).int()
        
        #Assuming connectivity is limited to nxn cube surrounding the node (n is odd)
        conn_window = 3
        z_c = torch.min(torch.max(z,torch.tensor(conn_window/2).int()),torch.tensor(dims[2]-1).int()-torch.tensor(conn_window/2).int())
        y_c = torch.min(torch.max(y,torch.tensor(conn_window/2).int()),torch.tensor(dims[1]-1).int()-torch.tensor(conn_window/2).int())
        x_c = torch.min(torch.max(x,torch.tensor(conn_window/2).int()),torch.tensor(dims[0]-1).int()-torch.tensor(conn_window/2).int())
        choice_neighbors = np.random.choice(conn_window**3, fanout, replace=False)
        choice_neighbors = torch.from_numpy(choice_neighbors)
        #Can we change it to torch.cat()?
        list_connected = []
        list_weights = []
        from_node_type = 1 - np.int32((node_types[i]+1)/2)
        for neighbor in choice_neighbors:
            z_loc = neighbor.clone().detach()//(conn_window**2) + z_c - 1
            y_loc = (neighbor % (conn_window**2)) // conn_window + y_c - 1
            x_loc = (neighbor % (conn_window**2)) % conn_window + x_c - 1
            neighbor_id = (z_loc*dims[0]*dims[1] + y_loc*dims[0] + x_loc).cpu().item()
            to_node_type = (1 - ((node_types[neighbor_id]+1)/2)).int().cpu().item()
            list_connected.append(neighbor_id)
            list_weights.append(w_matrix[from_node_type, to_node_type])
            adj_mat[i, neighbor_id] = w_matrix[from_node_type, to_node_type]
        all_connections.append(list_connected)
        all_weights.append(list_weights)
    return adj_mat, torch.tensor(all_connections), torch.tensor(all_weights)
#w_mat = np.array([[3, 6],[-2, -2]])
#adj_mat, adj_list, weight_list = create_random_reservoir((5,5,5), 0.2, w_mat, 9)