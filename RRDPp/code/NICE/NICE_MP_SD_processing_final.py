# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:24:45 2023

@author: Ida Olsen

Uses EASE-grid to produce 25 km grid mean values. Mean values are obtained either by the distance limit or the time limit of 30 days.
Uses input files magnaprobe measurements from the N-ICE2015 campaign 
includes Warren snow depths and densities
"""
# -- File info -- #

__author__ = 'Ida Olsen'
__contributors__ = 'Henriette Skorup'
__contact__ = ['ilo@dmi.dk']
__version__ = '1'
__date__ = '2025-08-08'

# -- Built-in modules -- #
import os.path
import sys
import glob
import re
import datetime as dt

# -- Third-part modules -- #
import numpy as np
from numpy.lib import recfunctions as rfn
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
#import PyPDF2
import pandas as pd
import csv

# -- Proprietary modules -- #
sys.path.append(os.path.dirname(os.getcwd()))
from Warren import SnowDepth, SWE
import EASEgrid_correct as EASEgrid
import Functions

#%% Main

# Information
name = 'NICE'
save_path = os.path.dirname(os.path.dirname(os.getcwd())) + f'/{name}/'
save_path_data = os.path.dirname(os.path.dirname(
    os.getcwd())) + f'/FINAL/{name}/final/'
if not os.path.exists(save_path_data):os.makedirs(save_path_data)
ofile = f'ESACCIplus-SEAICE-RRDP2+-SD-{name}.nc'
gridres = 25000  # grid resolution
dtint = 30  # days per mean
## saving locations
ofile = os.path.join(save_path_data, ofile)


saveplot = os.path.join(os.path.dirname(
    os.path.dirname(os.getcwd())), f'FINAL/{name}/fig/')
# create directory if they do not exist
if not os.path.exists(saveplot): os.makedirs(saveplot)
files = sorted(glob.glob(f'/dmidata/users/ilo/projects/RRDPp/RawData/{name}/Magmaprobe/N-ICE2015_MP_v1/*.txt'))

datalen = 0
count = 0  # used to locate first file

# Prepare empty lists to accumulate all files' data
all_lat = []
all_lon = []
all_SD = []
all_time = []

#defines variables in output file
count+=1
dataOut = Functions.Final_Data(Type='SD', count_head=count)
dataOut.pp_flag = 0

for ifile in files:
    Snow=False #default as most files does not contain snowdepth measurements
    file = os.path.basename(ifile)
    print(file)
    # values: [date/time (matlab)] [counter] [snow thickness in cm] [original latitude] [original longitude] [driftcorrected latitude] [driftcorrected longitude]
    names = ['date_matlab', 'counter', 'SD_cm', 'lat_orig', 'lon_orig', 'lat_corrected', 'lon_corrected']
    data = np.genfromtxt(ifile, names=names, dtype=None, delimiter=None, encoding=None)

    #print(data)
    latitude = data['lat_corrected']
    longitude = data['lon_corrected']
    SD = data['SD_cm']/100 # SD in m
    gps_date = data['date_matlab']
    t = [dt.datetime.fromordinal(int(gd)) + dt.timedelta(days=gd%1) - dt.timedelta(days=366) for gd in gps_date]

    # replace faulty values if applicable
    SD[SD < 0] = np.nan
    SD[SD > 2] = np.nan

    datalen += len(SD[np.isfinite(SD)])

    # Append current file's data to master lists
    all_lat.append(latitude)
    all_lon.append(longitude)
    all_SD.append(SD)
    all_time.append(t)

# Combine lists into final arrays
latitude = np.concatenate(all_lat)
longitude = np.concatenate(all_lon)
SD = np.concatenate(all_SD)
t = np.concatenate([np.array(times) for times in all_time])

## Define fixed uncertainty
SD_unc = np.ones(len(latitude)) * 0.003 #(from data source: https://data.npolar.no/dataset/3d72756d-788b-4c49-b0cc-8a345c091020)

# Define obsID
dataOut.obsID = 'N-ICE2015_MP'
dataOut.unc_flag = 3
dataOut.pp_flag = 0

# Create EASEgrid and returns grid cell indicies (index_i, index_j) for each observation
G = EASEgrid.Gridded()
G.SetHemisphere('N')
G.CreateGrids(gridres)
(index_i, index_j) = G.LatLonToIdx(latitude, longitude)

# Takes the time for each grid cell into account and calculate averages
avgSD, stdSD, lnSD, uncSD, lat, lon, time, avgSIT, stdSIT, lnSIT, uncSIT, avgFRB, stdFRB, lnFRB, FRB_Unc, var1, var2, dataOut.QFT, dataOut.QFS, dataOut.QFG = G.GridData(
    dtint, latitude, longitude, t, SD=SD, SD_unc=SD_unc, dtype='buoy')

if len(time) > 0:
    Functions.plot(latitude, longitude, dataOut.obsID, time,saveplot, HS='NH')
    Functions.scatter(dataOut.obsID, t, SD, time, avgSD, 'SD [m]', saveplot)


# Correlates HEM data with Warren snow depth and snow density
for ll in range(np.size(avgSD,0)):
    (w_SD,w_SD_epsilon) = SnowDepth(lat[ll],lon[ll],time[ll].month)
    dataOut.w_SD_final = np.append(dataOut.w_SD_final,w_SD)
    (wswe,wswe_epsilon) = SWE(lat[ll],lon[ll],time[ll].month)
    w_density=int((wswe/w_SD)*1000)
    dataOut.w_density_final = np.append(dataOut.w_density_final,w_density)

#Change names to correct format names
dataOut.obsID = [dataOut.obsID]*len(lat)
## pp flag + unc flag
dataOut.unc_flag = [dataOut.unc_flag]*len(lat)
dataOut.pp_flag = [dataOut.pp_flag]*len(lat)
dataOut.lat_final = lat
dataOut.lon_final = lon
for ll in range(np.size(time,0)):
    dataOut.date_final = np.append(dataOut.date_final,dt.datetime.strftime(time[ll],"%Y-%m-%dT%H:%M:%S"))
dataOut.time = [np.datetime64(d) for d in dataOut.date_final]
dataOut.SD_final = avgSD
dataOut.SD_std = stdSD
dataOut.SD_ln = lnSD
dataOut.SD_unc = uncSD
dataOut.SIT_final = avgSIT
dataOut.SIT_std = stdSIT
dataOut.SIT_ln = lnSIT
dataOut.SIT_unc = uncSIT

# fill empty arrays with NaN values
dataOut.Check_Output()
    
if count>1:
    subset = dataOut.Create_NC_file(ofile,primary='SD')
    df = Functions.Append_to_NC(df, subset)
else:
    df = dataOut.Create_NC_file(ofile,primary='SD', datasource='Magnaprobe, N-ICE2015, DOI:https://doi.org/10.21334/NPOLAR.2016.3D72756D', key_variables='Snow depth')

print(datalen)
# Save data to NetCDF
Functions.save_NC_file(df, ofile, primary='SD')

    

