import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset
from .folded_dataset import FoldedDataset

from options import opt


def get_msda_amazon_datasets(data_file, domain, kfold, feature_num):
    print(f'Loading mSDA Preprocessed Multi-Domain Amazon data for {domain} Domain')
    dataset = pickle.load(open(data_file, 'rb'))[domain]

    lx, ly = dataset['labeled']
    if feature_num > 0:
        lx = lx[:, : feature_num]
    lx = torch.from_numpy(lx.toarray()).float().to(opt.device)
    ly = torch.from_numpy(ly).long().to(opt.device)
    print(f'{domain} Domain has {len(ly)} labeled instances.')
    # if opt.use_cuda:
    #     lx, ly = lx.cuda(), ly.cuda()
    labeled_set = FoldedDataset(TensorDataset, kfold, lx, ly)

    ux, uy = dataset['unlabeled']
    if feature_num > 0:
        ux = ux[:, : feature_num]
    ux = torch.from_numpy(ux.toarray()).float().to(opt.device)
    uy = torch.from_numpy(uy).long().to(opt.device)
    print(f'{domain} Domain has {len(uy)} unlabeled instances.')
    # if opt.use_cuda:
    #     ux, uy = ux.cuda(), uy.cuda()
    unlabeled_set = TensorDataset(ux, uy)

    return labeled_set, unlabeled_set
