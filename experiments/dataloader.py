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
from model.manifold.sphere import Sphere



def latlon2xyz(lat,lon):
    lat = lat*np.pi/180
    lon = lon*np.pi/180 
    x= np.cos(lat)*np.cos(lon)
    y= np.cos(lat)*np.sin(lon)
    z= np.sin(lat)
    return x,y,z

def load_dataset(dataset_dir, batch_size, position_file, val_batch_size=None, include_context=False, **kwargs):

    if val_batch_size == None:
        val_batch_size = batch_size
    
    with open(dataset_dir + '/trn.pkl', 'rb') as f:
        data = pickle.load(f)
    with open(position_file, 'rb') as f:
        lonlat = pickle.load(f)['lonlat']

    train_numpy = data['x']
    feature_len = data['x'].shape[-1]

    kernel_generator = KernelGenerator(lonlat)
    kernel_info = {'sparse_idx': kernel_generator.sparse_idx,
                    'MLP_inputs': kernel_generator.MLP_inputs,
                    'geodesic': kernel_generator.geodesic.flatten(),
                    'angle_ratio':kernel_generator.ratio_lists.flatten()}
    
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

class KernelGenerator:
    def __init__(self, lonlat, k_neighbors=25, local_map='fast') -> None:
        self.lonlat = lonlat
        self.k_neighbors = k_neighbors
        self.local_map = local_map
        
        self.nbhd_idx, col, row, self.geodesic = self.get_geo_knn_graph(self.lonlat, self.k_neighbors)
        self.sparse_idx = np.array([row, col])
        self.MLP_inputs, self.centers, self.points = self.X2KerInput(self.lonlat, sparse_idx=self.sparse_idx, k_neighbors=self.k_neighbors, local_map=self.local_map)
        _, self.ratio_lists = self.XY2Ratio(self.MLP_inputs[:,-2:], k_neighbors=self.k_neighbors)
        
    def get_geo_knn_graph(self, X, k=25):
        #X: num_node, dim
        lon = X[:, 0]
        lat = X[:, 1]
        x,y,z = latlon2xyz(lat, lon)
        coordinate = np.stack([x,y,z])
        product = np.matmul(coordinate.T, coordinate).clip(min=-1.0, max=1.0) 
        geodesic = np.arccos(product)
        nbhd_idx = np.argsort(geodesic, axis=-1)[:,:k]
        col = nbhd_idx.flatten()
        row = np.expand_dims(np.arange(geodesic.shape[0]), axis=-1).repeat(k, axis=-1).flatten()
        return nbhd_idx, col, row, np.sort(geodesic, axis=-1)[:,:k]

    def X2KerInput(self, x, sparse_idx, k_neighbors, local_map='fast'):
        '''
        x: the location list of each point
        sparse_idx: the sparsity matrix of 2*num_nonzero
        '''
        sample_num = x.shape[0]
        loc_feature_num = x.shape[1]
        centers = x[sparse_idx[0]]
        points = x[sparse_idx[1]]
        if local_map == 'fast':
            delta_x = points - centers
            delta_x[delta_x>180] = delta_x[delta_x>180] - 360
            delta_x[delta_x<-180] = delta_x[delta_x<-180] + 360
            inputs = np.concatenate((centers, delta_x), axis=-1).reshape(-1, loc_feature_num*2)
            inputs = inputs/180*np.pi

        elif local_map == 'log':
            centers = torch.from_numpy(centers.reshape(-1, k_neighbors, loc_feature_num))
            points = torch.from_numpy(points.reshape(-1, k_neighbors, loc_feature_num))
            sphere_2d = Sphere(2)
            centers_x = torch.stack(Sphere.latlon2xyz(centers[:,0,1], centers[:,0,0]), dim=-1)
            points = torch.stack(Sphere.latlon2xyz(points[:,:,1], points[:,:,0]), dim=-1)
            log_cp = sphere_2d.log_map(centers_x, points)
            local_coor = sphere_2d.cart3d_to_tangent_local2d(centers_x, log_cp)
            
            centers = centers.reshape(-1, loc_feature_num).numpy()
            local_coor = local_coor.reshape(-1, loc_feature_num).numpy()
            inputs = np.concatenate((centers/180*np.pi, local_coor), axis=-1).reshape(-1, loc_feature_num*2)

        elif local_map == 'horizon':
            centers = torch.from_numpy(centers.reshape(-1, k_neighbors, loc_feature_num))
            points = torch.from_numpy(points.reshape(-1, k_neighbors, loc_feature_num))
            sphere_2d = Sphere(2)
            centers_x = torch.stack(Sphere.latlon2xyz(centers[:,0,1], centers[:,0,0]), dim=-1)
            points = torch.stack(Sphere.latlon2xyz(points[:,:,1], points[:,:,0]), dim=-1)
            h_cp = sphere_2d.horizon_map(centers_x, points)
            local_coor = sphere_2d.cart3d_to_ctangent_local2d(centers_x, h_cp)
            
            centers = centers.reshape(-1, loc_feature_num).numpy()
            local_coor = local_coor.reshape(-1, loc_feature_num).numpy()
            inputs = np.concatenate((centers/180*np.pi, local_coor), axis=-1).reshape(-1, loc_feature_num*2)
        else:
            raise NotImplementedError('The mapping is not provided.')
        
        return inputs, centers, points  
    
    def XY2Ratio(self, X, k_neighbors=25):
        x = X[:,0]
        y = X[:,1]
        thetas = np.arctan2(y,x)
        thetas = thetas.reshape(-1, k_neighbors)
        ratio_lists = []
        multiples = []
        for theta in thetas:
            theta_unique, counts = np.unique(theta, return_counts=True)
            multiple_list = np.array([theta_unique, counts]).T
            idx = np.argsort(theta_unique)
            multiple_list = multiple_list[idx]
            ratios = []
            ratios_theta = np.zeros_like(theta)
            for i in range(multiple_list.shape[0]):
                if i < multiple_list.shape[0] - 1:
                    ratio = (np.abs(multiple_list[i+1][0] - multiple_list[i][0]) + np.abs(multiple_list[i-1][0] - multiple_list[i][0]))/(2*2*np.pi)
                else: 
                    ratio = (np.abs(multiple_list[0][0]  - multiple_list[i][0]) + np.abs(multiple_list[i-1][0] - multiple_list[i][0]))/(2*2*np.pi)
                ratio = ratio/multiple_list[i][1]
                ratios.append(ratio)
                ratios_theta[theta == multiple_list[i][0]] = ratio
            ratio_lists.append(ratios_theta)
            multiple_list = np.concatenate([multiple_list, np.array([ratios]).T], axis=-1)
            multiples.append(multiple_list)
        return thetas, np.array(ratio_lists)
        
        
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
