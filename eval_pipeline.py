import torch_geometric
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import random
import os
from utils import create_channel_matrix, exhaustive_search, calc_rates, create_PyG_graph, gnn
import pathlib

# fixed global parameters
batch_size = 32 # batch size
num_epochs = 500 # number of epochs
lr = 1e-2 # learning rate
num_features_list = [1] + [64] * 3 # number of GNN features in different layers
BW = 10e6 # bandwidth (Hz)
N = -174 - 30 + 10 * np.log10(BW) # Noise PSD = -174 dBm/Hz
noise_var = np.power(10, N / 10) # noise variance
P_max = np.power(10, (10 - 30) / 10) # maximum transmit power = 10 dBm

def eval(model_state_dict, k, num_test_samples, random_seed):
    """
    evaluate a trained link scheduling GNN on networks with multiple transmitter-receiver pairs
    inputs:
        model_state_dict: PyTorch state dictionary containing the parameters of the trained model
        k: number of transmitter-receiver pairs for evaluation
        num_test_samples: number of test samples
        random_seed: random seed

    output:
        normalized test sum-rate
    """
    # set the random seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    k_range = [k]
    num_samples = {'test': num_test_samples}
    phases = list(num_samples.keys())

    current_file_path = str(pathlib.Path(__file__).parent.resolve())

    data_path = current_file_path + '/data/data_eval_k_{}_num_samples_{}.json'.format(k, num_test_samples)
    if not os.path.exists(data_path):
        # create datasets
        H = defaultdict(list)
        for phase in num_samples:
            for k in k_range:
                for _ in range(num_samples[phase]):
                    h = create_channel_matrix(k)
                    H[phase, k].append(h)
                H[phase, k] = torch.Tensor(np.stack(H[phase, k]))

        # create PyTorch Geometric datasets and dataloaders
        data_list = defaultdict(list)
        baseline_rates = dict()
        for phase in num_samples:
            for k in k_range:
                p_ex_search = exhaustive_search(H[phase, k], P_max, noise_var)
                for h, y in zip(H[phase, k], p_ex_search):
                    data_list[phase, k].append(create_PyG_graph(h, P_max, noise_var, y))
                # calculate baseline rates
                baseline_rates[phase, k, 'Exhaustive Search'] = torch.sum(calc_rates(H[phase, k], p_ex_search, P_max, noise_var), dim=1).detach().cpu().numpy()
        # save the generated data
        torch.save([data_list, baseline_rates], data_path)

    data_list, baseline_rates = torch.load(data_path)

    # dataloaders
    loader = {}
    for key in data_list:
        loader[key] = DataLoader(data_list[key], batch_size=batch_size, shuffle=('train' in key))

    # main training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    GNN = gnn(num_features_list).to(device)
    GNN.load_state_dict(model_state_dict)
    GNN.eval()
    phase = 'test'

    all_rates = []
    for data in loader[phase, k]:
        data = data.to(device)
        x, p_logits = GNN(data.x, data.edge_index, data.edge_weight)
        sum_rates = torch.sum(calc_rates(data.h, 1. * (p_logits > 0).view(-1, k), P_max, noise_var), dim=1)
        all_rates.extend(sum_rates.detach().cpu().numpy().tolist())

    return np.mean(all_rates) / np.mean(baseline_rates[phase, k, 'Exhaustive Search'])
