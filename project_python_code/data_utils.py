import matplotlib
from matplotlib import pyplot as plt
import os, os.path as osp
import json
import torch
from collections import defaultdict
import netCDF4 as nc
import numpy as np
import cv2

import ipdb
debug = ipdb.set_trace
def read_ncdf_precip(path):
    # precip(time, lat, lon)
    ds = nc.Dataset(path)
    datadict=  {}
    datadict['precip'] = np.clip(np.array(ds['precip']), 0, None) # clip because lowest value is super low
    datadict['lat'] = np.array(ds['lat'])
    datadict['lon'] = np.array(ds['lon'])
    datadict['time'] = np.array(ds['time'])
    return datadict

def read_ncdf_temp(path):
    ds = nc.Dataset(path)
    datadict = {}
    datadict['lat'] = np.array(ds['latitude'])
    datadict['lon'] = np.array(ds['longitude'])
    datadict['time'] = np.array(ds['time'])
    datadict['t2m'] = np.array(ds['t2m'])  - 273.0 # convert to C
    return datadict

def read_ncdf_pressure(path):
    ds = nc.Dataset(path)
    datadict = {}
    datadict['lat'] = np.array(ds['latitude'])
    datadict['lon'] = np.array(ds['longitude'])
    datadict['time'] = np.array(ds['time'])
    datadict['pressure'] = np.array(ds['z'])  
    return datadict

def get_input_tensors_temporal(N, res, sigma=2.0, sigma_small=0.5):
    Z_base = torch.normal(mean=0, std=sigma, size=(1,  )+tuple(res))
    z_inputs = [Z_base,]
    for i in range(N-1):
        z_inputs.append(torch.normal(mean=0, std=sigma+sigma_small, size=( 1,)+tuple(res)))

    return z_inputs


def get_guidance_tensor(idx, size):
    data = read_ncdf_temp("/home/kv30/KV/climate/data/raw/2m_temperature_2017.nc")
    precip_tensor = data['t2m'][idx,...]
    print(precip_tensor.shape, data['t2m'].shape)
    plt.figure()
    plt.imshow(precip_tensor, cmap='magma')
    plt.show()
    guide = cv2.resize(precip_tensor, size)
    return guide