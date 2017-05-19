'''
This file describes the process for computing weighted climate data
'''

import xarray as xr
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.ndimage import label
from scipy.interpolate import griddata
from six import string_types
import itertools
import glob
import toolz
import os

WEIGHTS_FILE = os.path.join(
    '/shares/gcp/climate/_spatial_data/world-combo-new/segment_weights',
    'agglomerated-world-new_BCSD_grid_segment_weights_area_pop.csv')



'''
=================
Private Functions
=================
'''

def _fill_holes_xr(
        ds,
        varname,
        broadcast_dims=('time',),
        lon_name='lon',
        lat_name='lat',
        gridsize=0.25,
        minlat=-85,
        maxlat=85):
    '''
    Fill NA values inplace in a gridded dataset
    
    Parameters
    ----------
    
    ds : xarray.Dataset
        name of the dataset with variable to be modified
    
    varname : str
        name of the variable to be interpolated
    
    broadcast_dims : tuple of strings, optional
        tuple of dimension names to broadcast the interpolation step over
        (default 'time')
    
    lon_name : str, optional
        name of the longitude dimension (default 'lon')

    lat_name : str, optional
        name of the latitude dimension (default 'lat')
        
    gridsize : float, optional
        size of the lat/lon grid. Important for creating a bounding box around
        NaN regions (default 0.25)
    
    minlat : float, optional
        latitude below which no values will be interpolated (default -85)
    
    minlon : float, optional
        latitude above which no values will be interpolated (default 85)
    
    '''
    if isinstance(broadcast_dims, string_types):
        broadcast_dims = (broadcast_dims, )

    ravel_lons, ravel_lats = (
        np.meshgrid(ds.coords[lon_name].values, ds.coords[lat_name].values))
    
    # remove infinite values
    ds[varname] = ds[varname].where((ds[varname] < 1e30))
    
    for indexers in itertools.product(*tuple(
            [range(len(ds.coords[c])) for c in broadcast_dims])):

        slicer_dict = dict(zip(broadcast_dims, indexers))
        
        slicer = tuple([
                slicer_dict[c]
                if c in broadcast_dims
                else slice(None, None, None)
                for c in ds[varname].dims])

        sliced = ds[varname].values.__getitem__(slicer)
        
        if not np.isnan(sliced).any():
            continue

        filled = _fill_holes(
            var=np.ma.masked_invalid(sliced),
            lat=ravel_lats,
            lon=ravel_lons,
            gridsize=0.25,
            minlat=-85,
            maxlat=85)
        
        ds[varname][slicer_dict] = filled


def _fill_holes(var, lat, lon, gridsize=0.25, minlat=-85, maxlat=85):
    '''
    Interpolates the missing values between points on grid

    Parameters
    ----------
    var: masked np.array
        array of climate values 

    lat: masked np.array 
        array of latitude values

    lon: masked np. array
        array of longitude values

    gridsize: int
        corresponds to degrees on the grid for climate data 

    minlat: int
        corresponds to min latitude values to include. Used to remove poles

    maxlat: int
        corresponds to max lat values to include. Used to remove poles


    '''
    # fill the missing value regions by linear interpolation
    # pass if no missing values
    if not np.ma.is_masked(var):
        return var
    # or the missing values are only in polar regions
    if not var.mask[20: -20, :].any():
        return var

    # fill the holes
    var_filled = var[:]
    missing = np.where((var.mask == True) & (lat > minlat) & (lat < maxlat))
    mp = np.zeros(var.shape)
    mp[missing] = 1
    ptch, n_ptch = label(mp)

    for p in range(1, n_ptch+1):

        ind_ptch = np.where(ptch == p)
        lat_ptch = lat[ind_ptch]
        lon_ptch = lon[ind_ptch]

        ind_box = np.where(
                (lat <= np.max(lat_ptch)+gridsize) &
                (lat >= np.min(lat_ptch)-gridsize) &
                (lon <= np.max(lon_ptch)+gridsize) &
                (lon >= np.min(lon_ptch)-gridsize))

        var_box = var[ind_box]
        lat_box = lat[ind_box]
        lon_box = lon[ind_box]
        not_missing = np.where(var_box.mask==False)
        points = np.column_stack([lon_box[not_missing], lat_box[not_missing]])
        values = var_box[var_box.mask==False]
        var_filled[ind_box] = griddata(
                points,
                values,
                (lon_box, lat_box),
                method='linear')

    return var_filled


def _standardize_longitude_dimension(ds, lon_names=['lon', 'longitude']):
    '''
    Rescales the lat and lon coordinates to ensure lat is within (-90,90) 
    and lon is within (-180, 180). Renames coordinates 
    from lon to longitude and from lat to latitude. Sorts any new
    rescaled coordinated. 

    Parameters
    ----------
    ds: xarray.DataSet

    Returns
    -------
    ds: xarray.DataSet

    .. note:: this will be unnecessary if we standardize inputs. We can
    scale the longitude dim to between (-180, 180)

    '''

    coords = np.array(ds.coords.keys())

    assert len(coords[np.in1d(coords, lon_names)]) == 1
    _lon_coord = coords[np.in1d(coords, ['longitude', 'lon'])][0]

    ds = ds.rename({_lon_coord: '_longitude'})

    # Adjust lat and lon to make sure they are within (-90, 90) and (-180, 180)
    ds['_longitude_adjusted'] = (
        (ds._longitude - 360)
            .where(ds._longitude > 180)
            .fillna(ds._longitude))

    # reassign the new coords to as the main lon coords
    ds = (ds
        .swap_dims({'_longitude': '_longitude_adjusted'})
        .reindex({'_longitude_adjusted': sorted(ds._longitude_adjusted)}))

    if '_longitude' in ds.coords:
        ds = ds.drop('_longitude')

    ds = ds.rename({'_longitude_adjusted': _lon_coord})

    return ds


@toolz.memoize
def _prepare_spatial_weights_data(weights_file=WEIGHTS_FILE):
    '''
    Rescales the pix_cent_x colum values

    Parameters
    ----------
    weights_file: str
        location of file used for weighting


    .. note:: unnecessary if we can standardize our input
    '''

    df = pd.read_csv(weights_file)

    # Re-label out-of-bounds pixel centers
    df.set_value((df['pix_cent_x'] == 180.125), 'pix_cent_x', -179.875)

    #probably totally unnecessary
    df.drop_duplicates()
    df.index.names = ['reshape_index']

    df.rename(
        columns={'pix_cent_x': 'lon', 'pix_cent_y': 'lat'},
        inplace=True)

    return df


def _reindex_spatial_data_to_regions(da, df):
    '''
    Reindexes spatial and segment weight data to regions

    Enables region index-based math operations

    Parameters
    ----------
    da: Xarray DataArray
    df: Pandas DataFrame

    Returns
    -------
    Xarray DataArray


    ''' 
    res = da.sel_points(
        'reshape_index', 
        lat=df.lat.values, 
        lon=df.lon.values)

    return res


def _aggregate_reindexed_data_to_regions(
        da,
        variable,
        weights_df,
        region_id_string,
        backup_variable='areawt'):
    '''
    Performs weighted avg for climate variable by region

    Parameters
    ----------

    da: xarray.DataArray

    variable: str
        variable to weight by (i.e popwt, areawt, cropwt)

    weight_df: pd.DataFrame
        pandas DataFrame of weights

    region_id_string: str
        indicates which regional id scheme to select in the dataframe

    backup_variable: str
        if no variable is provided will default to `areawt`

    '''

    da.coords[region_id_string] = xr.DataArray(
                weights_df[region_id_string].values,
                dims={'reshape_index': weights_df.index.values})

    # get backup weights
    da.coords[backup_variable] = xr.DataArray(
                weights_df[backup_variable].values,
                dims={'reshape_index': weights_df.index.values})

    ds = da.to_dataset().reset_coords(backup_variable)

    weights_backup = (
                ds[backup_variable]
                    .groupby(region_id_string)
                    .sum(dim='reshape_index'))

    # get preferred wieghts
    da.coords[variable] = xr.DataArray(
                weights_df[variable].values, dims={'reshape_index':
                weights_df.index.values})

    ds = da.to_dataset().reset_coords(variable)

    weights_preferred = (
                ds[variable]
                    .groupby(region_id_string)
                    .sum(dim='reshape_index'))

    weights = (
                weights_preferred
                    .where(weights_backup > 0)
                    .fillna(weights_backup))

    weighted = (
                (ds['tasmin']*ds[variable])
                    .groupby(region_id_string)
                    .sum(dim='reshape_index')/weights)

    return weighted


'''
================
Public Functions
================
'''

def load_climate_data(fp, varname, lon_name='lon'):
    '''
    Read and prepare climate data

    After reading data, this method also fills NA values using linear
    interpolation, and standardizes longitude to -180:180

    Parameters
    ----------
    fp: str
        File path to dataset

    varname: str
        Variable name to be read

    lon_name : str, optional
        Name of the longitude dimension (defualt selects from ['lon' or
        'longitude'])

    Returns
    -------
    xr.Dataset
         xarray dataset loaded into memory 
    '''

    if lon_name is not None:
        lon_names = [lon_name]

    with xr.open_dataset(fp) as ds:

        _fill_holes_xr(ds.load(), varname)
        return _standardize_longitude_dimension(ds, lon_names=lon_names)


def weighted_aggregate_grid_to_regions(
        data,
        socio_variable,
        region_id,
        weights_file=WEIGHTS_FILE):
    '''
    Computes the weighted reshape of gridded data

    Parameters
    ----------
    data: xr.DataArray
        xarray DataArray to be aggregated. Must have 'lat' and 'lon' in the
        coordinates.

    socio_variable: str
        Weighting variable (e.g. 'popwt', 'areawt'). This must be a column name
        in the weights file.

    region_id: str
        Target regional aggregation level (e.g. 'ISO', 'hierid'). This must be
        a column name in the weights file.

    weights_file: str, optional
        Path to the file used for weighting (default agglomerated-world-new
        BCSD segment weights)

    Returns
    -------
    ds: xr.Dataset
        weighted and averaged dataset based on region_id
    '''

    region_weights = _prepare_spatial_weights_data(weights_file)
    rdxd = _reindex_spatial_data_to_regions(data, region_weights)
    wtd = _aggregate_reindexed_data_to_regions(rdxd, socio_variable, df, region_id)
    return wtd

