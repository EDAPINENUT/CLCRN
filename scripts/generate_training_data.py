from typing import Text
from numpy.core.overrides import ArgSpec
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd
import sys
import pickle
from pathlib import Path
import torch
from xarray.core.variable import Coordinate
np.random.seed(2020)

def latlon2xyz(lat,lon):
    lat = lat*np.pi/180
    lon = lon*np.pi/180 
    x= np.cos(lat)*np.cos(lon)
    y= np.cos(lat)*np.sin(lon)
    z= np.sin(lat)
    return x,y,z


def date_to_inputseq(time_data, mask_dataset, args):
    tmpdata = time_data[mask_dataset]
    horizon = args.output_horizon_len
    input_len = args.input_seq_len
    step_size = args.step_size
    L = tmpdata.shape[0]
    dataset_year=[]
    dataset_month=[]
    dataset_day=[]
    dataset_hour=[]
    for i in range(input_len):
        dataset_year.append(pd.DatetimeIndex(np.array(tmpdata[i:L-horizon-input_len+i])).year)
        dataset_month.append(pd.DatetimeIndex(np.array(tmpdata[i:L-horizon-input_len+i])).month)
        dataset_day.append(pd.DatetimeIndex(np.array(tmpdata[i:L-horizon-input_len+i])).day)
        dataset_hour.append(pd.DatetimeIndex(np.array(tmpdata[i:L-horizon-input_len+i])).hour)

    dataset_year = np.stack(dataset_year,axis=1)
    dataset_month = np.stack(dataset_month,axis=1)
    dataset_day = np.stack(dataset_day,axis=1)
    dataset_hour = np.stack(dataset_hour, axis=1)

    num_samples, input_len = dataset_year.shape[0], dataset_year.shape[1]

    idx = [i*step_size for i in range(num_samples//step_size)]

    dataset_year, dataset_month, dataset_day, dataset_hour = dataset_year[idx], dataset_month[idx], dataset_day[idx], dataset_hour[idx]
    
    
    dataset_time = np.stack([dataset_year, dataset_month, dataset_day, dataset_hour], axis=-1)
    # dataset_time = np.stack([dataset_month, dataset_hour], axis=-1)
    
    return dataset_time

def dataset_to_seq2seq(raw_data, mask_dataset, args):
    tmpdata = raw_data[mask_dataset]
    print('Mean:{}, Max{}, Min{}, Std{}'.format(tmpdata.mean().values, tmpdata.max().values, tmpdata.min().values, tmpdata.std().values))
    horizon = args.output_horizon_len
    input_len = args.input_seq_len
    step_size = args.step_size
    L = tmpdata.shape[0]
    dataset_x=[]
    dataset_y=[]
    for i in range(input_len):
        dataset_x.append(tmpdata[i:L-horizon-input_len+i])
    for j in range(horizon):
        dataset_y.append(tmpdata[j+input_len:L-horizon+j])
    dataset_x = np.stack(dataset_x,axis=1)
    dataset_y = np.stack(dataset_y,axis=1)
    num_samples, input_len = dataset_x.shape[0], dataset_x.shape[1]
    dataset_x = dataset_x.reshape(num_samples, input_len, -1)
    dataset_y = dataset_y.reshape(num_samples, horizon, -1)
    idx = [i*step_size for i in range(num_samples//step_size)]
    dataset_x = dataset_x[idx]
    dataset_y = dataset_y[idx]
    return dataset_x, dataset_y

def main(args):
    for ii in range(len(args.datasets)):
        
        constants = xr.open_mfdataset(args.raw_dataset_dir +'/{}/*.nc'.format('constants'), combine='by_coords')
        lsm = np.array(constants.lsm).flatten()
        height = np.array(constants.orography).flatten()
        latitude = np.array(constants.lat2d).flatten()
        longitude = np.array(constants.lon2d).flatten()
        geo_context = np.stack([lsm, height, latitude, longitude], axis=-1)

        if args.datasets[ii] != 'component_of_wind':
            data = xr.open_mfdataset(args.raw_dataset_dir +'/{}/*.nc'.format(args.datasets[ii]), combine='by_coords')
            time = data.time.values
            mask_dataset = np.bitwise_and(np.datetime64(args.start_date)<=time , time<=np.datetime64(args.end_date))
            lon,lat = np.meshgrid(data.lon-180, data.lat)
            lonlat = np.array([lon, lat])
            lonlat = lonlat.reshape(2,32*64).T
            raw_data = data.__getattr__(args.attri_names[ii])

            if len(raw_data.shape) == 4: # when there are different level, we choose the 13-th level which is sea level
                raw_data = raw_data[:,-1,...]
            seq2seq_data, seq2seq_label = dataset_to_seq2seq(raw_data, mask_dataset, args)

            time_data = data.time
            time_context = date_to_inputseq(time_data, mask_dataset, args)
            
            num_samples = seq2seq_data.shape[0]
            node_num = geo_context.shape[0]
            time_len = time_context.shape[1]
            
            time_context = np.repeat(time_context[:,:,None,:], node_num, axis=2)
            geo_context = np.repeat(geo_context[None,:,:], time_len * num_samples, axis=0).reshape(num_samples, time_len, node_num, -1)
            context = np.concatenate([time_context, geo_context], axis=-1)

            num_test = round(num_samples * 0.2)
            num_train = round(num_samples * 0.7)
            num_val = num_samples - num_test - num_train
            print('Number of training samples: {}, validation samples:{}, test samples:{}'.format(num_train, num_val, num_test))

            if args.shuffle:
                idx = np.random.permutation(np.arange(num_samples))
                seq2seq_data = seq2seq_data[idx]
                context = context[idx]
                seq2seq_label = seq2seq_label[idx]

            train_x = seq2seq_data[:num_train][:,:,:,None]
            train_context = context[:num_train]
            train_y = seq2seq_label[:num_train][:,:,:,None]
            
            val_x = seq2seq_data[num_train:num_train+num_val][:,:,:,None]
            val_context = context[num_train:num_train+num_val]
            val_y = seq2seq_label[num_train:num_train+num_val][:,:,:,None]

            test_x = seq2seq_data[num_train+num_val:][:,:,:,None]
            test_context = context[num_train+num_val:]
            test_y = seq2seq_label[num_train+num_val:][:,:,:,None]
            datasets =[[train_x, train_y, train_context], [val_x, val_y, val_context], [test_x,test_y, test_context]]
            subsets = ['trn','val','test']
            path = args.output_dirs[ii]
            path_ = Path(path)
            path_.mkdir(exist_ok=True,parents=True)
            
            for i, subset in enumerate(subsets):
                with open(path+'/{}.pkl'.format(subset), "wb") as f:
                    save_data = {'x': datasets[i][0],
                                'y': datasets[i][1],
                                'context': datasets[i][2]}
                    pickle.dump(save_data,f, protocol = 4)
            with open(path+'/{}.pkl'.format('position_info'), "wb") as f:
                save_data = {'lonlat': lonlat}
                pickle.dump(save_data, f, protocol = 4)

        else:
            data_u = xr.open_mfdataset(args.raw_dataset_dir + '/10m_u_component_of_wind/*.nc', combine='by_coords')
            data_v = xr.open_mfdataset(args.raw_dataset_dir + '/10m_v_component_of_wind/*.nc', combine='by_coords')
            
            time = data_u.time.values
            mask_dataset = np.bitwise_and(np.datetime64(args.start_date)<=time , time<=np.datetime64(args.end_date))
            lon,lat = np.meshgrid(data_u.lon-180, data_u.lat)
            lonlat = np.array([lon, lat])
            lonlat = lonlat.reshape(2,32*64).T

            raw_data_u = data_u.u10
            raw_data_v = data_v.v10

            seq2seq_data_u, seq2seq_label_u = dataset_to_seq2seq(raw_data_u, mask_dataset, args)
            seq2seq_data_v, seq2seq_label_v = dataset_to_seq2seq(raw_data_v, mask_dataset, args)

            time_data = data.time
            time_context = date_to_inputseq(time_data, mask_dataset, args)
            
            num_samples = seq2seq_data_u.shape[0]
            node_num = geo_context.shape[0]
            time_len = time_context.shape[1]
            
            time_context = np.repeat(time_context[:,:,None,:], node_num, axis=2)
            geo_context = np.repeat(geo_context[None,:,:], time_len * num_samples, axis=0).reshape(num_samples, time_len, node_num, -1)
            context = np.concatenate([time_context, geo_context], axis=-1)
            
            num_test = round(num_samples * 0.2)
            num_train = round(num_samples * 0.7)
            num_val = num_samples - num_test - num_train
            print('Number of training samples: {}, validation samples:{}, test samples:{}'.format(num_train, num_val, num_test))
            
            if args.shuffle:
                idx = np.random.permutation(np.arange(num_samples))
                seq2seq_data_u = seq2seq_data_u[idx]
                seq2seq_label_u = seq2seq_label_u[idx]
                seq2seq_data_v = seq2seq_data_v[idx]
                seq2seq_label_v = seq2seq_label_v[idx]
                context = context[idx]

            train_x = np.stack([seq2seq_data_u[:num_train], seq2seq_data_v[:num_train]], axis=-1)
            train_y = np.stack([seq2seq_label_u[:num_train],seq2seq_label_v[:num_train]], axis=-1)
            val_x = np.stack([seq2seq_data_u[num_train:num_train+num_val], seq2seq_data_v[num_train:num_train+num_val]], axis=-1)
            val_y = np.stack([seq2seq_label_u[num_train:num_train+num_val],seq2seq_label_v[num_train:num_train+num_val]], axis=-1)
            test_x = np.stack([seq2seq_data_u[num_train+num_val:], seq2seq_data_v[num_train+num_val:]], axis=-1)
            test_y = np.stack([seq2seq_label_u[num_train+num_val:],seq2seq_label_v[num_train+num_val:]], axis=-1)
            
            train_context = context[:num_train]
            val_context = context[num_train:num_train+num_val]
            test_context = context[num_train+num_val:]

            datasets =[[train_x, train_y, train_context], [val_x, val_y, val_context], [test_x, test_y, test_context]]
            subsets = ['trn','val','test']
            path = args.output_dirs[ii]
            path_ = Path(path)
            path_.mkdir(exist_ok=True,parents=True)
            for  i, subset in enumerate(subsets):
                with open(path+'/{}.pkl'.format(subset), "wb") as f:
                    save_data = {'x': datasets[i][0],
                                'y': datasets[i][1],
                                'context': datasets[i][2]}
                    pickle.dump(save_data,f, protocol = 4)
            with open(path+'/{}.pkl'.format('position_info'), "wb") as f:
                save_data = {'lonlat': lonlat}
                pickle.dump(save_data, f, protocol = 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dataset_dir",
        type=str,
        default='dataset_release/WeatherBench/datasets'
    )
    parser.add_argument(
        "--datasets", type=list, default=[
            '2m_temperature',
            'relative_humidity', 
            'component_of_wind', 
            'total_cloud_cover'
            ], help="dataset name."
    )
    parser.add_argument(
        "--attri_names", type=list, default=['t2m', 'r','uv10','tcc'], help="data name."
    )
    parser.add_argument(
        "--output_dirs", type=list, default=['data/temperature','data/humidity', 'data/component_of_wind','data/cloud_cover'], help="Output directory."
    )
    parser.add_argument(
        "--step_size", type=int, default=24
    )
    parser.add_argument(
        "--input_seq_len", type=int, default=12
    )
    parser.add_argument(
        "--output_horizon_len", type=int, default=12
    )
    parser.add_argument(
        '--start_date', type=str, default='2010-01-01'
    )
    parser.add_argument(
        '--end_date', type=str, default='2019-01-01'
    )
    parser.add_argument(
        "--shuffle", type=bool, default=False
    )
    args = parser.parse_args()
    main(args)