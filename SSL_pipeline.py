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
from utils import create_channel_matrix, exhaustive_search, calc_rates, create_PyG_graph, gnn, augment_ITLQ
import pathlib

# fixed global parameters
batch_size = 32 # batch size
num_SSL_epochs = 100 # number of epochs
lr = 1e-2 # learning rate
num_features_list = [1] + [64] * 3 # number of GNN features in different layers
BW = 10e6 # bandwidth (Hz)
N = -174 - 30 + 10 * np.log10(BW) # Noise PSD = -174 dBm/Hz
noise_var = np.power(10, N / 10) # noise variance
P_max = np.power(10, (10 - 30) / 10) # maximum transmit power = 10 dBm
tau = 0.1 # temperature hyperparameter for constrastive SSL
eta_augment_ITLQ = 0.5 # ITLinQ parameter for identifying weak interference links during the augmentation process

def pretrain_SSL(k, num_samples, random_seed):
    """
    pre-train a link scheduling GNN backbone on networks with multiple transmitter-receiver pairs using contrastive self-supervised learning
    inputs:
        k: number of transmitter-receiver pairs
        num_samples: dictionary containing number of train/test samples
        mode: training mode ('Supervised'/'Unsupervised')
        random_seed: random seed

    outputs:
        all_epoch_results: dictionary containing the results of all epochs
        best_model: dictionary containing the parameters of the final pre-trained GNN
    """

    # set the random seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    k_range = [k]
    phases = list(num_samples.keys())

    current_file_path = str(pathlib.Path(__file__).parent.resolve())

    data_path = current_file_path + '/data/data_k_{}_num_samples_{}.json'.format(k, str(num_samples))
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
    optimizer = torch.optim.Adam(list(GNN.parameters()), lr=lr)
    criterion_CE = torch.nn.CrossEntropyLoss()

    all_epoch_results = defaultdict(list)
    best_model = None
    for epoch in tqdm(range(num_SSL_epochs)):

        for phase in phases:
            if 'train' in phase:
                GNN.train()
            else:
                GNN.eval()

            all_rates = []
            all_losses = []
            all_accs = []
            for data in loader[phase, k]:

                optimizer.zero_grad()

                data = data.to(device)

                # track history if only in train
                with torch.set_grad_enabled('train' in phase):

                    # create two augmentations of the batch of channel gains
                    h_aug = [augment_ITLQ(data.h, P_max, noise_var, eta_augment_ITLQ) for _ in range(2)]
                    data_list_aug = [[create_PyG_graph(h, P_max, noise_var) for h in aug] for aug in h_aug]
                    batch_aug = [next(iter(DataLoader(list_aug, batch_size=len(list_aug), shuffle=False))) for list_aug in data_list_aug]

                    # contrastive SSL
                    embeddings_aug0, _ = GNN(batch_aug[0].x, batch_aug[0].edge_index, batch_aug[0].edge_weight)
                    embeddings_aug1, _ = GNN(batch_aug[1].x, batch_aug[1].edge_index, batch_aug[1].edge_weight)

                    # normalize the embeddings to the unit hypersphere
                    embeddings_aug0 = F.normalize(embeddings_aug0, dim=1)
                    embeddings_aug1 = F.normalize(embeddings_aug1, dim=1)

                    all_embeddings = torch.cat((embeddings_aug0, embeddings_aug1), dim=0)

                    logits = torch.matmul(all_embeddings, torch.transpose(all_embeddings, 0, 1)) / tau - \
                                1e10 * torch.eye(len(all_embeddings)).to(device) # ignore the diagonals

                    B = len(embeddings_aug0)
                    labels = list(np.arange(B, 2 * B)) + list(np.arange(B))
                    labels = torch.Tensor(labels).to(device=device, dtype=torch.long)

                    loss = criterion_CE(logits, labels)

                    if 'train' in phase:
                        # Backward pass
                        loss.backward()
                        optimizer.step()

                    all_losses.append(loss.item())

            # save average epoch results
            all_epoch_results[phase, k, 'loss'].append(np.mean(all_losses))

        # save the model if it is the best performing so far
        if all_epoch_results['test', k, 'loss'][-1] == np.min(all_epoch_results['test', k, 'loss']):
            best_model = GNN.state_dict()

    return all_epoch_results, best_model
