import numpy as np
import torch
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import LEConv
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse
from scipy.spatial import distance

# fixed global parameters
R = 250 # network area side length
min_D_TxTx = 35 # minimum Tx-Tx distance
min_D_TxRx = 10 # minimum Tx-Rx distance
max_D_TxRx = 50 # maximum Tx-Rx distance
shadowing = 7 # shadowing standard deviation

def UDN_PL(D, alpha1=2):
    m, n = np.shape(D)
    L = np.zeros((m, n))
    k0 = 39
    a1 = alpha1
    a2 = 4
    db = 100

    CONST = 10 * np.log10(db ** (a2-a1))

    for i in range(m):
        for j in range(n):
            d = D[i,j]
            if d <= db:
                L[i,j] = k0 + 10 * a1 * np.log10(d)
            else:
                L[i,j] = k0 + 10 * a2 * np.log10(d) - CONST
    return L

def create_channel_matrix(k):
    # specify transmitter locations
    while True:
        locTx = np.random.uniform(0, R, (k, 2)) - R / 2
        D_TxTx = distance.cdist(locTx, locTx, 'euclidean')
        for Tx in range(k):
            D_TxTx[Tx, Tx] = float('Inf')
        if np.min(D_TxTx) >= min_D_TxTx:
            break

    # specify receiver locations
    phi = 2 * np.pi * np.random.uniform(k)
    r = np.sqrt(np.random.uniform(low=min_D_TxRx ** 2, high=max_D_TxRx ** 2, size=(k,)))
    locRx = np.clip(locTx + np.stack((r * np.cos(phi), r * np.sin(phi)), axis=1), a_min= -R / 2, a_max=R / 2)

    D_TxRx = distance.cdist(locTx, locRx, 'euclidean')

    L = UDN_PL(D_TxRx) + shadowing * np.random.randn(k, k) # Loss matrix in dB
    H_l = np.sqrt(np.power(10, -L / 10)) # large-scale fading matrix

    return np.abs(H_l) ** 2

def calc_rates(h, p, P_max, noise_var):
    """
    calculate rates for a batch of b networks, each with k transmitter-reciever pairs
    inputs:
        h: b x k x k tensor containing a batch of k x k channel matrices
        p: b x k tensor containing normalized transmit power levels (0: no transmission; 1: full transmit power)
        P_max: scalar indicating maximum transmit power
        noise_var: scalar indicating noise variance

    output:
        rates: b x k tensor containing user rates
    """
    b = h.shape[0]
    k = p.shape[1]

    signal = p * torch.diagonal(h, dim1=1, dim2=2)
    interference = torch.sum(p.unsqueeze(-1).repeat(1, 1, k) * h, dim=1) - signal

    rates = torch.log2(1 + signal / (interference + noise_var / P_max))

    return rates

def exhaustive_search(h, P_max, noise_var):
    """
    derive sum-rate optimal power levels for a batch of b networks, each with k transmitter-reciever pairs via exhaustive search
    inputs:
        h: b x k x k tensor containing a batch of k x k channel matrices
        P_max: scalar indicating maximum transmit power
        noise_var: scalar indicating noise variance

    output:
        p: b x k tensor containing optimal normalized binary transmit power levels (0: no transmission; 1: full transmit power), which lead to best sum-rates

    """
    b, k, _ = h.shape
    all_possible_binary_power_level_vectors = torch.Tensor(np.array(list(itertools.product([0, 1], repeat=k))[1:])) # ignore the first row (all zeros)
    all_sum_rates = []
    for p_vec in all_possible_binary_power_level_vectors:
        p = p_vec.unsqueeze(0).repeat(b, 1)
        rates = calc_rates(h, p, P_max, noise_var)
        sum_rates = torch.sum(rates, dim=1)
        all_sum_rates.append(sum_rates)
    all_sum_rates = torch.stack(all_sum_rates, dim=1)
    optimal_p = all_possible_binary_power_level_vectors[torch.argmax(all_sum_rates, dim=1)]

    return optimal_p

def create_PyG_graph(h, P_max, noise_var, y=None):
    h_log = (torch.log(P_max * h / noise_var))
    h_log[h_log == -float("Inf")] = 0
    h_norm = torch.norm(h_log)
    edge_index, edge_weight = from_scipy_sparse_matrix(sparse.csr_matrix(h_log / h_norm))
    x = torch.log(P_max * torch.diag(h) / noise_var).unsqueeze(1) / h_norm
    if y is not None:
        y = y.unsqueeze(1)
    PyG_graph = Data(x=x,
                     y=y,
                     edge_index=edge_index,
                     edge_weight=edge_weight,
                     h=h.unsqueeze(0),
                    )
    return PyG_graph

class gnn(torch.nn.Module):
    def __init__(self, num_features_list):
        super(gnn, self).__init__()
        num_layers = len(num_features_list)
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(LEConv(num_features_list[i], num_features_list[i + 1]))
        self.PC_head = nn.Linear(num_features_list[-1], 1)

    def forward(self, x, edge_index, edge_weight):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.leaky_relu(x)
        p_logits = self.PC_head(x)
        return x, p_logits

def augment_ITLQ(h, P_max, noise_var, eta, jitter_max=0.1):
    h_copy = np.copy(h.detach().cpu().numpy())
    h_copy *= np.random.uniform(low=1-jitter_max, high=1+jitter_max, size=h_copy.shape)
    B, k, _ = h_copy.shape
    g = P_max / noise_var

    output = np.zeros_like(h_copy)
    for sample in range(B):
        H = h_copy[sample]
        for i in range(k):
            for j in np.setdiff1d(range(k), i):
                if g * H[i, j] < (g * min(H[i, i], H[j, j])) ** eta: # weak interference link
                    H[i, j] *= np.random.binomial(n=1, p=0.5)
        output[sample] = H
    return torch.Tensor(output)
