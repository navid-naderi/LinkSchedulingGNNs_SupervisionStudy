# Wireless Link Scheduling via Graph Representation Learning: A Comparative Study of Different Supervision Levels

This repository contains the source code for implementing models based on graph neural networks (GNNs) to solve link scheduling problems in wireless networks for sum-rate maximization. The code includes different supervision levels, including supervised, unsupervised, and self-supervised training procedures, and compares models trained with different supervision levels/types in terms of system-level sum-rate, convergence behavior, sample efficiency, and generalization capability. Please refer to [the accompanying paper](https://arxiv.org/abs/2110.01722) for more details.

The GNN architecture used in this repo is based on the [PyTorch Geometric implementation](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.LEConv) of the local extremum operator, termed LEConv, from the “[ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations](https://ojs.aaai.org/index.php/AAAI/article/view/5997/5853)” paper at AAAI 2020. The GNN architecture can be easily modified to any of the several GNN architectures implemented in PyTorch Geometric that support edge weights (see the table [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html)).

## Dependencies

* [PyTorch](https://pytorch.org/)
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [tqdm](https://tqdm.github.io/)
* [Matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)

## Citation

Please use the following BibTeX citation if you use this repository in your work:

```
@article{LinkSchedulingGNNs_SupervisionStudy_naderializadeh2021,
  title={Wireless Link Scheduling via Graph Representation Learning: A Comparative Study of Different Supervision Levels},
  author={Naderializadeh, Navid},
  journal={arXiv preprint arXiv:2110.01722},
  year={2021}
}
```
