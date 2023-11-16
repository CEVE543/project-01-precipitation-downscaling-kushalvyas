# %%
import matplotlib
from matplotlib import pyplot as plt
import os, os.path as osp
import json
from collections import defaultdict
import netCDF4 as nc
from netCDF4 import num2date, date2num, date2index

import numpy as np

import ipdb
debug = ipdb.set_trace

# %%
year = 2019
gt_data_file = "../data/gt/precip_tx.nc"
inp_data_file = f"../data/raw/Houston_latlon_res_02/2m_temperature_{year}.nc"
inp_pressure_file = f"../data/raw/Houston_latlon_res_02/500hPA_geopotential_{year}.nc"

# %%
def read_ncdf_precip(path):
    # precip(time, lat, lon)
    ds = nc.Dataset(path)
    datadict=  {}
    datadict['precip'] = np.clip(np.array(ds['precip']), 0, None) # clip because lowest value is super low
    datadict['lat'] = np.array(ds['lat'])
    datadict['lon'] = np.array(ds['lon'])
    datadict['time'] = ds["time"]
    return datadict

def read_ncdf_temp(path):
    ds = nc.Dataset(path)
    datadict = {}
    datadict['lat'] = np.array(ds['latitude'])
    datadict['lon'] = np.array(ds['longitude'])
    datadict['time'] = ds["time"]
    datadict['t2m'] = np.array(ds['t2m'])  - 273.0 # convert to C
    return datadict

def read_ncdf_pressure(path):
    ds = nc.Dataset(path)
    datadict = {}
    datadict['lat'] = np.array(ds['latitude'])
    datadict['lon'] = np.array(ds['longitude'])
    datadict['time'] = ds["time"]
    datadict['pressure'] = np.array(ds['z'])  
    return datadict
# %%
gt_data = read_ncdf_precip(gt_data_file)

# %%
inp_data = read_ncdf_temp(inp_data_file)
# %%
inp_pressure = read_ncdf_pressure(inp_pressure_file)
# %%
idx=8682
from datetime import datetime
dates = num2date(gt_data['time'][:], gt_data['time'].units)

plt.figure()
plt.title(dates[idx])
plt.imshow(gt_data["precip"][idx,...])
plt.colorbar();
plt.set_cmap('magma')
plt.show()


# %%
plt.figure()
plt.title(dates[idx])
plt.imshow(inp_data["t2m"][idx,...])
plt.colorbar();
plt.set_cmap('magma')
plt.show()
# %%
