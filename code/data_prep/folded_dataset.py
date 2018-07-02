import pickle
import random
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset

from options import opt

class FoldedDataset:
    def __init__(self, baseclass, kfold, *args):
        self.__class__ = type(self.__class__.__name__,
                (baseclass, object),
                dict(self.__class__.__dict__))
        super(self.__class__, self).__init__(*args)

        self.kfold = kfold
        self.split_folds(kfold)
        print('Loaded {}-fold dataset of {} samples'.format(kfold, len(self)))

    def split_folds(self, kfold):
        n_samples = len(self)
        if not opt.use_preshuffle:
            random.seed(opt.random_seed)
            np.random.seed(opt.random_seed)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            pickle.dump(indices,
                    open(os.path.join(opt.model_save_file, 'shuffle-indices.pkl'), 'wb'))
        else:
            indices = pickle.load(open(opt.amazon_preshuffle_file, 'rb'))
        fold_sizes = (n_samples // kfold) * np.ones(kfold, dtype=np.int)
        fold_sizes[:n_samples % kfold] += 1
        current = 0
        self.folds = {}
        for fold, fold_size in enumerate(fold_sizes):
            start, stop = current, current + fold_size
            self.folds[fold] = indices[start:stop]
            current = stop

    def get_folds(self, folds):
        indices = np.hstack([self.folds[f] for f in folds]).reshape(-1)
        if self.__class__.__bases__[0].__name__ == 'TensorDataset':
            indices = torch.from_numpy(indices).to(opt.device)
            # if opt.use_cuda:
            #     indices = indices.cuda()
            X = torch.index_select(self.tensors[0], 0, indices)
            Y = torch.index_select(self.tensors[1], 0, indices)
            return TensorDataset(X, Y)
        else:
            X = [self.X[i] for i in indices]
            indices = torch.from_numpy(indices).to(opt.device)
            # if opt.use_cuda:
            #     indices = indices.cuda()
            Y = torch.index_select(self.Y, 0, indices)
        return AmazonDataset(X, Y, self.max_seq_len)

    def get_trainset(self, fold):
        if fold+1 == self.kfold:
            folds = list(range(1, self.kfold-1))
        else:
            folds = list(range(fold)) + list(range(fold+2, self.kfold))
        return self.get_folds(folds)

    def get_devset(self, fold):
        folds = [fold]
        return self.get_folds(folds)
    
    def get_testset(self, fold):
        folds = [(fold+1) % self.kfold]
        return self.get_folds(folds)

