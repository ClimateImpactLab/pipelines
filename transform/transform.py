#Prototype interface for climate transformations
##################################################
#Given a job specification, do_thing will perform a transformation
#and weighted average of the socio variable. It is just two functions. 
#We can specify an entire universe of jobs and then dispatch them wherever.
########################################
#Job specification can be automated with something like the following
#These paremeters below are really just for file path specification
#The only part that will require effort on the ra is the definitation of 
#the transformations. These functions can then be shipped out on a per-job basis


#this config setup will generate 108 unique jobs. 
#
#import itertools
#
# config_elements  = {
#     'weights_path': ['BCSD', 'GMFD', 'BEST'],#weights
#     'b': ['USA', 'FRA', 'JAP'],#regions
#     'c': [lambda x: x**2, lambda x: x**3, lambda x: x**4],#transformations
#     'd': ['tas', 'prcp'],#clim_var
#     'e': ['A'],#resample period
#     'f': ['mean'],#resample method
#     'g': ['popwt', 'crop'], #socio var
#     'h': ['hierid'], #region_id
#     'i': ['areawt'] #backup_socio  
# }

#product = [x for x in apply(itertools.product, config_elements.values())]
#job_list = [dict(zip(config_elements.keys(), p)) for p in product]

# By appending the jobs to a list then dispatching individual jobs in parallel
###############################


import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
import dask.array as da
import glob
import time
import datafs
import ipyparallel


def _standardize_longitude_dimension(ds):
    '''
    Rescales the lat and lon coordinates to ensure lat is within (-90,90) 
    and lon is within (-180, 180). Renames coordinates 
    from lon to longitude and from lat to latitude. Sorts any new
    rescaled coordinated. 

    Parameters
    ----------
    ds: object
    `< Xarray Dataset http://xarray.pydata.org/en/stable/\
    data-structures.html#dataset>`

    Returns
    -------
    ds: object
    `Xarray Dataset`__ 

    __ http://xarray.pydata.org/en/stable/data-structures.html#dataset

    .. note:: this will be unnecessary if we standardize inputs. We can
    scale the longitude dim to between (-180, 180)

    '''

    t1 = time.time()
    coords = ds.coords
    #name checking, coerce lat and lon to latitude and longitude
    if 'longitude' not in coords:
        ds = ds.rename({'lon': 'longitude'})

    if 'latitude' not in coords:
        ds = ds.rename({'lat': 'latitude'})

    #Adjust lat and lon to make sure they are within (-90, 90) and (-180, 180)
    ds['longitude_adjusted'] = (ds.longitude - 360).where(
                                ds.longitude > 180).fillna(ds.longitude)
    # ds['latitude_adjusted'] = (ds.latitude - 180).where(
    #                             ds.latitude > 90).fillna(ds.latitude)
    ds['latitude_adjusted'] = ds.latitude

    #reassign the new coords to as the main lon and lat coords
    ds = ds.swap_dims(
                    {'longitude': 'longitude_adjusted'}
                    ).reindex(
                    {'longitude_adjusted': sorted(ds.longitude_adjusted)})

    ds = ds.swap_dims(
                    {'latitude': 'latitude_adjusted'
                    }).reindex(
                    {'latitude_adjusted': sorted(ds.latitude_adjusted)})


    t2 = time.time()
    print('_standardize_long_dim: {}'.format(t2 - t1))

    return ds

def _rescale_reshape_weights(df):
    '''
    Rescales the pix_cent_x colum values
    Some of the values are greater than 180 and we want to constrain values
    to (-180, 180)

    Parameters
    ----------
    df: Pandas DataFrame

    .. note:: unnecessary if we can standardize our input
    '''

    t1 = time.time() 
    df.set_value((df['pix_cent_x'] == 180.125), 'pix_cent_x', -179.875)
    #probably totally unnecessary
    df.drop_duplicates()
    df.index.names = ['reshape_index']
    df.rename(columns={
        'pix_cent_x': 'longitude_adjusted', 
        'pix_cent_y': 'latitude_adjusted'
        }, 
        inplace=True)

    t2 = time.time()
    print('_rescale_reshape_weights: {}'.format(t2 - t1))


    
    return df

def _reindex(da, df):
    '''
    Reshapes and rescales climate data and segment weighted data along the 
    same axis.
    Enables index based math operations.

    Parameters
    ----------
    da: Xarray DataArray
    df: Pandas DataFrame

    Returns
    -------
    Xarray DataArray


    ''' 
    tic = time.time()

    res = da.sel_points('reshape_index', 
                        latitude_adjusted=df.latitude_adjusted.values, 
                        longitude_adjusted=df.longitude_adjusted.values)

    toc = time.time()
    print('_reindex.sel_points: {}'.format(toc - tic))

    return res
    

def transform(path, variable, transformation, frequency, how='mean'):
    '''
    Arbitrary function to perform on variable. 

    Parameters
    ----------
    path: str
        path to file to be opened

    variable: str
        'tas', 'prcp', 

    function: func
        Some transformation to perform on 
        Examples: edd, binning, polynomial, etcs

    frequency: str
        capitalized first letter of annual, monthly or daily depending
        on time resolution. (i.e. 'A', 'M', 'D')
        Full list of offset aliases in `here`__

        __ http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases 

    how: str
        'mean' or 'sum'

    Returns
    -------
    `XArrray DataArray`__

    __ http://xarray.pydata.org/en/stable/data-structures.html#dataset

    '''

    ds = xr.open_dataset(path).load()

    tic = time.time()
    tr = transformation(ds[variable])
    toc = time.time()
    print('TRANSFORM: {}'.format(toc - tic))

    t1 = time.time()
    res = tr.resample(frequency, 'time', how=how, keep_attrs=True)
    t2 = time.time()
    print('RESAMPLE: {}'.format(t2 - t1))

    return res



def weighted_avg(data_array, variable, weights, region_id_string, backup_variable='areawt'):
    '''
    Performs weighted avg for climate variable by region

    Parameters
    ----------

    data_array: XArray object that corresponds to the climate variable

    variable: str
        passed as argument to the weights dataframe

    weights: Pandas DataFrame 
        Sector segment weights by geography

    region_id_string: str
        column in Pandas DataFrame to sum along

    backup_variable: str
        Replacement if variable is zero or nan

    '''
    t1 = time.time()
    weighted_ds = weights.to_xarray()

    t2 = time.time()
    print('weighted_avg.to_xarray: {}'.format(t2 - t1))

    t1 = time.time()
    weighted_ds.coords[region_id_string] = weighted_ds[region_id_string]

    weights_preferred = (weighted_ds[variable].groupby(
                region_id_string).sum(dim='reshape_index'))

    weights_backup = (weighted_ds[backup_variable].groupby(
                region_id_string).sum(dim='reshape_index'))

    weights = weights_preferred.where(weights_backup > 0).fillna(
        weights_backup)

    weighted = (data_array*weighted_ds[variable]).groupby(region_id_string).sum(
                dim='reshape_index')/weights


    t2 = time.time()
    print('weighted_avg.math: {}'.format(t2 - t1))

    return weighted



def to_datafs(archive_name, type='csv',  output_location=None):
    '''
    Writes an archive to DataFS with output of transformation. 

    Parameters
    ----------
    type: 'str'
        file type extension. i.e. 'csv', 'nc'

    archive_name: str
        name of variable to be stored in datafs

    local_dir: str
        If you want to save it locally, you can provide  a file extension
        and it wll save a file.
    '''
    pass

def write_to_netcdf(data, path):

    t1 = time.time()
    if path is not None:
        data.to_netcdf(path)

    t2 = time.time()
    print('file written to {}'.format(path))
    print('write_to_netcdf: {}'.format(t2 - t1))




def do_thing(job):

    df = pd.read_csv(job['weights_path'])
    weights = _rescale_reshape_weights(df) 
    
    t1 = time.time()
    transformed = transform(job['clim_path'], job['clim_var'],
                        job['transformation'], job['resample_period'], 
                        how=job['resample_method'])

    write_to_netcdf(transformed, job['clim_transformed_output_dir'])

    standardized = _standardize_longitude_dimension(transformed)
    reindexed = _reindex(standardized, weights)

    reshaped = weighted_avg(reindexed, job['socio_variable'], weights,
                             job['region_id'], 
                             job['backup_socio_var'])
    tic = time.time()
    write_to_netcdf(reshaped, job['weighted_variable_output_dir'])
    toc = time.time()
    print('to_netcdf: {}'.format(toc - tic))
    t2 = time.time()

