import torch
from pytracking.features.net_wrappers import NetWithBackbone


atom_net = NetWithBackbone(net_path='ml_mb10_atom_p05_005_prop128',
                             use_gpu=False, initialize=True)


dimp18_net = NetWithBackbone(net_path='dimp18',
                             use_gpu=False, initialize=True)


kldimp18_net = NetWithBackbone(net_path='dimp18_klce_wlr_bbw01',
                             use_gpu=False, initialize=True)


def compare_nets(net1, net2):
    err = 0
    for p1, p2 in zip(net1.feature_extractor.parameters(), net2.feature_extractor.parameters()):
        err = err + (p1 - p2).abs().sum()
    return err.item()


print(compare_nets(atom_net, dimp18_net))
print(compare_nets(atom_net, kldimp18_net))
print(compare_nets(dimp18_net, kldimp18_net))