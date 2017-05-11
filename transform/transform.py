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


    #Path /shares/gcp/sources/BCSD-original/{}/day/atmos/tasmax/r1i1p1/v1.0/tasmax_day_BCSD_rcp85_r1i1p1_{}_{}: rcp, model,year
###############################


import xarray as xr
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.ndimage import label
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import seaborn 
from six import string_types
import itertools

import glob
import time



def fill_holes_xr(ds, varname, broadcast_dims=('time',), lon_name='lon', lat_name='lat', gridsize=0.25, minlat=-85, maxlat=85):
    '''
    Fill NA values inplace in a gridded dataset
    
    Parameters
    ----------
    
    ds : xarray.Dataset
        name of the dataset with variable to be modified
    
    varname : str
        name of the variable to be interpolated
    
    broadcast_dims : tuple of strings, optional
        tuple of dimension names to broadcast the interpolation step over (default 'time')
    
    lon_name : str, optional
        name of the longitude dimension (default 'lon')

    lat_name : str, optional
        name of the latitude dimension (default 'lat')
        
    gridsize : float, optional
        size of the lat/lon grid. Important for creating a bounding box around NaN regions (default 0.25)
    
    minlat : float, optional
        latitude below which no values will be interpolated (default -85)
    
    minlon : float, optional
        latitude above which no values will be interpolated (default 85)
    
    '''
    
    if isinstance(broadcast_dims, string_types):
        broadcast_dims = (broadcast_dims, )

    ravel_lons, ravel_lats = np.meshgrid(ds.coords[lon_name].values, ds.coords[lat_name].values)
    
    # remove infinite values
    ds[varname] = ds[varname].where((ds[varname] < 1e30))
    
    for indexers in itertools.product(*tuple([range(len(ds.coords[c])) for c in broadcast_dims])):
        slicer_dict = dict(zip(broadcast_dims, indexers))
        
        slicer = tuple([slicer_dict[c] if c in broadcast_dims else slice(None, None, None) for c in ds[varname].dims])
        sliced = ds[varname].values.__getitem__(slicer)
        
        if not np.isnan(sliced).any():
            continue

        filled = fill_holes(
            var=np.ma.masked_invalid(sliced),
            lat2=ravel_lats,
            lon2=ravel_lons,
            gridsize=0.25,
            minlat=-85,
            maxlat=85)
        
        ds[varname][slicer_dict] = filled
        
    
def fill_holes(var, lat2, lon2, gridsize=0.25, minlat=-85, maxlat=85):
    # fill the missing value regions by linear interpolation
    # pass if no missing values
    if not np.ma.is_masked(var):
        return var

    # fill the holes
    var_filled = var[:]
    missing = np.where((var.mask == True) & (lat2 > minlat) & (lat2 < maxlat))
    mp = np.zeros(var.shape)
    mp[missing] = 1
    ptch, n_ptch = label(mp)
    for p in range(1, n_ptch+1):
        ind_ptch = np.where(ptch == p)
        lat_ptch = lat2[ind_ptch]
        lon_ptch = lon2[ind_ptch]
        ind_box = np.where((lat2 <= np.max(lat_ptch)+gridsize) & (lat2 >= np.min(lat_ptch)-gridsize) & (lon2 <= np.max(lon_ptch)+gridsize) & (lon2 >= np.min(lon_ptch)-gridsize))
        var_box = var[ind_box]
        lat_box = lat2[ind_box]
        lon_box = lon2[ind_box]
        not_missing = np.where(var_box.mask==False)
        points = np.column_stack([lon_box[not_missing], lat_box[not_missing]])
        values = var_box[var_box.mask==False]
        var_filled[ind_box] = griddata(points, values, (lon_box, lat_box), method='linear')
    return var_filled
    


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

    df.set_value((df['pix_cent_x'] == 180.125), 'pix_cent_x', -179.875)
    #probably totally unnecessary
    df.drop_duplicates()
    df.index.names = ['reshape_index']
    df.rename(columns={
        'pix_cent_x': 'longitude_adjusted', 
        'pix_cent_y': 'latitude_adjusted'
        }, 
        inplace=True)
    
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

    res = da.sel_points('reshape_index', 
                        latitude_adjusted=df.latitude_adjusted.values, 
                        longitude_adjusted=df.longitude_adjusted.values)


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

    tr = transformation(ds[variable])

    res = tr.resample(frequency, 'time', how=how, keep_attrs=True)

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
    weighted_ds = weights.to_xarray()


    weighted_ds.coords[region_id_string] = weighted_ds[region_id_string]

    weights_preferred = (weighted_ds[variable].groupby(
                region_id_string).sum(dim='reshape_index'))

    weights_backup = (weighted_ds[backup_variable].groupby(
                region_id_string).sum(dim='reshape_index'))

    weights = weights_preferred.where(weights_backup > 0).fillna(
        weights_backup)

    weighted = (data_array*weighted_ds[variable]).groupby(region_id_string).sum(
                dim='reshape_index')/weights



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

    if path is not None:
        data.to_netcdf(path)





def do_thing(job):
    '''

    1. Reshape weighting dataframe
    2. Transform climate data by some function and resample along time axis
    3. Write transformed climate data to disk
    4. Align climate and socio data
    5. Get weighted average
    6. Write to disk

    '''

    df = pd.read_csv(job['weights_path'])
    weights = _rescale_reshape_weights(df) 
    
    transformed = transform(job['clim_path'], job['clim_var'],
                        job['transformation'], job['resample_period'], 
                        how=job['resample_method'])

    write_to_netcdf(transformed, job['clim_transformed_output_dir'])

    standardized = _standardize_longitude_dimension(transformed)
    reindexed = _reindex(standardized, weights)

    reshaped = weighted_avg(reindexed, job['socio_variable'], weights,
                             job['region_id'], 
                             job['backup_socio_var'])
    write_to_netcdf(reshaped, job['weighted_variable_output_dir'])

