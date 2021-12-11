import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import matplotlib.pyplot as plt
import torch
from scipy.sparse import linalg
from torch.utils.data import Dataset
from lib.utils import StandardScaler


def load_dataset(dataset_dir, batch_size, kernel_file, val_batch_size=None, include_context=False, **kwargs):

    if val_batch_size == None:
        val_batch_size = batch_size

    with open(dataset_dir + '/trn.pkl', 'rb') as f:
        data = pickle.load(f)
    with open(kernel_file, 'rb') as f:
        kernel_info = pickle.load(f)

    train_numpy = data['x']
    feature_len = data['x'].shape[-1]

    scaler = [StandardScaler(mean=train_numpy[..., i].mean(), std=train_numpy[..., i].std()) for i in range(feature_len)]
    
    train_set = Dataset(
        dataset_dir, mode='trn', batch_size=batch_size, scaler=scaler, include_context=include_context
    )

    validation_set = Dataset(
        dataset_dir, mode='val', batch_size=val_batch_size, scaler=scaler, include_context=include_context
    )

    test_set = Dataset(
        dataset_dir, mode='test', batch_size=val_batch_size, scaler=scaler, include_context=include_context
    )

    data = {}
    data['train_loader'] = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    data['val_loader']  = torch.utils.data.DataLoader(validation_set, batch_size=val_batch_size, shuffle=False, num_workers=4)
    data['test_loader'] = torch.utils.data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=4)
    data['scaler'] = scaler
    data['kernel_info'] = kernel_info

    return data

class Dataset(Dataset):
    """IRDataset dataset."""

    def __init__(self, dataset_dir, mode, batch_size, scaler=None, include_context=False):
        self.file = dataset_dir
        print('loading data of {} set...'.format(mode))
        with open(dataset_dir + '/{}.pkl'.format(mode), 'rb') as f:
            data = pickle.load(f)
        self.x = data['x']
        self.y = data['y']
        self.include_context = include_context

        if include_context == True:
            self.context = data['context']

        feature_len = data['x'].shape[-1]

        if scaler != None:
            self.scaler = scaler
            for i in range(feature_len):
                self.x[..., i] = scaler[i].transform(self.x[..., i])
                self.y[..., i] = scaler[i].transform(self.y[..., i])
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        data = self.x[idx]
        labels = self.y[idx]

        if self.include_context == True:
            data_context = self.context[idx]
            data = np.concatenate([data, data_context], axis=-1)

        return data, labels
