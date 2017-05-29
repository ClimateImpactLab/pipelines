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
import toolz
import os
import datafs
import click
import dill
import json
import pipelines

WEIGHTS_FILE = (
    'GCP/spatial/world-combo-new/segment_weights/' +
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
    missing = np.where((var.mask) & (lat > minlat) & (lat < maxlat))
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
        not_missing = np.where(var_box.mask == False)
        points = np.column_stack([lon_box[not_missing], lat_box[not_missing]])
        values = var_box[var_box.mask == False]
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

    api = datafs.get_api()
    archive = api.get_archive(weights_file)

    with archive.open('r') as f:
        df = pd.read_csv(f)

    # Re-label out-of-bounds pixel centers
    df.set_value((df['pix_cent_x'] == 180.125), 'pix_cent_x', -179.875)

    #probably totally unnecessary
    df.drop_duplicates()
    df.index.names = ['reshape_index']

    df.rename(
        columns={'pix_cent_x': 'lon', 'pix_cent_y': 'lat'},
        inplace=True)

    return df


def _reindex_spatial_data_to_regions(ds, df):
    '''
    Reindexes spatial and segment weight data to regions

    Enables region index-based math operations

    Parameters
    ----------
    ds: xarray Dataset
    df: pandas DataFrame

    Returns
    -------
    Xarray DataArray


    '''
    res = ds.sel_points(
        'reshape_index',
        lat=df.lat.values,
        lon=df.lon.values)

    return res


def _aggregate_reindexed_data_to_regions(
        ds,
        variable,
        aggwt,
        agglev,
        weights,
        backup_aggwt='areawt'):
    '''
    Performs weighted avg for climate variable by region

    Parameters
    ----------

    ds: xarray.DataArray

    variable: str
        name of the data variable

    aggwt: str
        variable to weight by (i.e popwt, areawt, cropwt)

    agglev: str
        indicates which regional id scheme to select in the dataframe

    weight: pd.DataFrame
        pandas DataFrame of weights

    backup_aggwt: str, optional
        aggregation weight to use in regions with no aggwt data (default
        'areawt')

    '''

    ds.coords[agglev] = xr.DataArray(
                weights[agglev].values,
                dims={'reshape_index': weights.index.values})

    # format weights
    ds[aggwt] = xr.DataArray(
                weights[aggwt].values,
                dims={'reshape_index': weights.index.values})

    ds[aggwt].where(ds[aggwt] > 0).fillna(weights[backup_aggwt].values)

    weighted = xr.Dataset({
        variable: (
            (ds[variable]*ds[aggwt])
                .groupby(agglev)
                .sum(dim='reshape_index') /
            ds[aggwt]
                .groupby(agglev)
                .sum(dim='reshape_index'))})

    return weighted


'''
================
Public Functions
================
'''

def load_climate_data(fp, varname, lon_name='lon', broadcast_dims=('time',)):
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

        _fill_holes_xr(ds.load(), varname, broadcast_dims=broadcast_dims)
        return _standardize_longitude_dimension(ds, lon_names=lon_names)


def weighted_aggregate_grid_to_regions(
        ds,
        variable,
        aggwt,
        agglev,
        weights=None):
    '''
    Computes the weighted reshape of gridded data

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset to be aggregated. Must have 'lat' and 'lon' in the
        coordinates.

    variable : str
        name of the variable to be aggregated

    aggwt : str
        Weighting variable (e.g. 'popwt', 'areawt'). This must be a column name
        in the weights file.

    agglev : str
        Target regional aggregation level (e.g. 'ISO', 'hierid'). This must be
        a column name in the weights file.

    weights : str, optional
        Regional aggregation weights (default agglomerated-world-new BCSD
        segment weights)

    Returns
    -------
    ds: xr.Dataset
        weighted and averaged dataset based on agglev
    '''

    if weights is None:
        weights = _prepare_spatial_weights_data()

    ds = _reindex_spatial_data_to_regions(ds, weights)
    ds = _aggregate_reindexed_data_to_regions(
        ds,
        variable,
        aggwt,
        agglev,
        weights)

    return ds


class bcsd_transform(object):

    @staticmethod
    @toolz.memoize
    def create_dummy_data(tmp, variable, **kwargs):

        tmp_path_in = os.path.join(tmp, 'sample_in.nc')

        time = pd.date_range('1/1/1981', periods=4, freq='3M')
        lats = np.arange(-89.875, 90, 0.25)
        lons = np.arange(-179.875, 180, 0.25)

        ds = xr.Dataset({
            variable: xr.DataArray(
                np.random.random((len(time), len(lats), len(lons))),
                dims=('time', 'lat', 'lon'),
                coords={
                    'time': time,
                    'lat': lats,
                    'lon': lons})
            })

        ds.to_netcdf(tmp_path_in)

        return {'read_file': tmp_path_in}

    @staticmethod
    @toolz.memoize
    def create_dummy_data_small(tmp, variable, **kwargs):

        tmp_path_in = os.path.join(tmp, 'sample_in_small.nc')

        time = pd.date_range('1/1/1981', periods=4, freq='3M')
        lats = np.arange(-0.875, 5, 1)
        lons = np.arange(20.875, 25, 1)

        ds = xr.Dataset({
            variable: xr.DataArray(
                np.random.random((len(time), len(lats), len(lons))),
                dims=('time', 'lat', 'lon'),
                coords={
                    'time': time,
                    'lat': lats,
                    'lon': lons})
            })

        ds.to_netcdf(tmp_path_in)

        return {'read_file': tmp_path_in}

    @staticmethod
    def run(
            read_file,
            write_file,
            variable,
            transformation,
            metadata,
            rcp,
            pername,
            years,
            model,
            agglev,
            aggwt,
            weights=None):

        # Load pickled transformation
        transformation = pipelines.load_func(transformation)

        # Add to job metadata
        metadata.update(dict(
            time_horizon='{}-{}'.format(years[0], years[-1])))

        # Get transformed data
        ds = xr.Dataset({variable: xr.concat([
            (load_climate_data(
                    read_file.format(year=y),
                    variable,
                    broadcast_dims=('time',))
                .pipe(transformation))
            for y in years],
            dim=pd.Index(years, name='year')).mean(dim='year')})
        
        # Reshape to regions
        if not agglev.startswith('grid'):
            ds = weighted_aggregate_grid_to_regions(
                    ds, variable, aggwt, agglev, weights=weights)

        # Update netCDF metadata
        ds.attrs.update(**metadata)

        # Write output
        if not os.path.isdir(os.path.dirname(write_file)):
            os.makedirs(os.path.dirname(write_file))

        ds.to_netcdf(write_file)

    @classmethod
    def run_test_small(cls, *args, **kwargs):
        weights = pd.read_pickle('pipelines/climate/test/data/weightsfile.pkl')
        with xr.open_dataset(kwargs['read_file']) as ds:
            weights = weights.loc[
                np.in1d(weights['lat'], ds.lat) &
                np.in1d(weights['lon'], ds.lon)]

        cls.run(*args, weights=weights, **kwargs)

    @classmethod
    def run_test(cls, *args, **kwargs):
        weights = pd.read_pickle('pipelines/climate/test/data/weightsfile.pkl')
        cls.run(*args, weights=weights, **kwargs)


class bcsd_transform_annual(bcsd_transform):

    @staticmethod
    def run(
            read_file,
            write_file,
            variable,
            transformation,
            transformation_name,
            metadata,
            rcp,
            pername,
            years,
            model,
            agglev,
            aggwt,
            weights=None):

        # print(read_file)
        # print(write_file)
        #print(metadata)
        for y in years:
            print(read_file.format(year=y))
            wf = write_file.format(year=y)
            print(wf)

                # Load pickled transformation
            transformation = pipelines.load_func(transformation)

            # Get transformed data
            ds = xr.Dataset(load_climate_data(
                        read_file.format(year=y),
                        variable,
                        broadcast_dims=('time',))
                    .pipe(transformation))
        
        # Reshape to regions
            if not agglev.startswith('grid'):
                ds = weighted_aggregate_grid_to_regions(
                        ds, variable, aggwt, agglev, weights=weights)

            # Update netCDF metadata
            ds.attrs.update(**metadata)

            # Write output
            if not os.path.isdir(os.path.dirname(write_file.format(
                            agglev=agglev, 
                            rcp=rcp, 
                            variable=variable, 
                            transformation_name=transformation_name))):
                os.makedirs(os.path.dirname(write_file.format(
                            agglev=agglev, 
                            rcp=rcp, 
                            variable=variable, 
                            transformation_name=transformation_name)))

            ds.to_netcdf(wf)





class pattern_transform(object):

    @staticmethod
    @toolz.memoize
    def create_dummy_data(tmp, variable, **kwargs):

        days = np.arange(1, 30)
        lats = np.arange(-89.875, 90, 0.25)
        lons = np.arange(-179.875, 180, 0.25)

        tmp_path_in = os.path.join(tmp, 'sample_in.nc')

        ds = xr.Dataset({
            variable: xr.DataArray(
                np.random.random((len(days), len(lats), len(lons))),
                dims=('day', 'lat', 'lon'),
                coords={
                    'day': days,
                    'lat': lats,
                    'lon': lons})
            })

        ds.to_netcdf(tmp_path_in)

        tmp_baseline = os.path.join(tmp, 'sample_baseline.nc')

        ds = xr.Dataset({
            variable: xr.DataArray(
                np.random.random((len(lats), len(lons))),
                dims=('lat', 'lon'),
                coords={
                    'lat': lats,
                    'lon': lons})
            })

        ds.to_netcdf(tmp_baseline)

        return {'pattern_file': tmp_path_in, 'baseline_file': tmp_baseline}

    @staticmethod
    @toolz.memoize
    def create_dummy_data_small(tmp, variable, **kwargs):

        days = np.arange(1, 3)
        lats = np.arange(-0.875, 5, 1)
        lons = np.arange(20.875, 25, 1)

        tmp_path_in = os.path.join(tmp, 'sample_in_small.nc')

        ds = xr.Dataset({
            variable: xr.DataArray(
                np.random.random((len(days), len(lats), len(lons))),
                dims=('day', 'lat', 'lon'),
                coords={
                    'day': days,
                    'lat': lats,
                    'lon': lons})
            })

        ds.to_netcdf(tmp_path_in)

        tmp_baseline = os.path.join(tmp, 'sample_baseline_small.nc')

        ds = xr.Dataset({
            variable: xr.DataArray(
                np.random.random((len(lats), len(lons))),
                dims=('lat', 'lon'),
                coords={
                    'lat': lats,
                    'lon': lons})
            })

        ds.to_netcdf(tmp_baseline)

        return {'pattern_file': tmp_path_in, 'baseline_file': tmp_baseline}

    @staticmethod
    def run(
            pattern_file,
            baseline_file,
            write_file,
            metadata,
            variable,
            transformation,
            rcp,
            pername,
            years,
            model,
            baseline_model,
            season,
            agglev,
            aggwt,
            weights=None):

        # Load pickled transformation
        transformation = pipelines.load_func(transformation)

        # Add to job metadata
        metadata.update(dict(
            time_horizon='{}-{}'.format(years[0], years[-1])))

        # Get transformed data
        ds = xr.Dataset({variable: xr.concat([
            (load_climate_data(
                    pattern_file.format(year=y),
                    variable,
                    broadcast_dims=('day',))
                .pipe(transformation))
            for y in years],
            dim=pd.Index(years, name='year')).mean(dim='year')})

        # load baseline
        with xr.open_dataset(baseline_file) as base:
            ds = (ds + base).load()

        # Reshape to regions
        if not agglev.startswith('grid'):
            ds = weighted_aggregate_grid_to_regions(
                    ds, variable, aggwt, agglev, weights=weights)

        # Update netCDF metadata
        ds.attrs.update(**metadata)

        # Write output
        if not os.path.isdir(os.path.dirname(write_file)):
            os.makedirs(os.path.dirname(write_file))

        ds.to_netcdf(write_file)

    @classmethod
    def run_test_small(cls, *args, **kwargs):
        weights = pd.read_pickle('pipelines/climate/test/data/weightsfile.pkl')
        with xr.open_dataset(kwargs['pattern_file']) as ds:
            weights = weights.loc[
                (np.in1d(weights['lat'].values, ds.lat) &
                                np.in1d(weights['lon'].values, ds.lon))].copy()

        cls.run(*args, weights=weights, **kwargs)

    @classmethod
    def run_test(cls, *args, **kwargs):
        weights = pd.read_pickle('pipelines/climate/test/data/weightsfile.pkl')
        cls.run(*args, weights=weights, **kwargs)


@click.command()
@click.argument('command')
@click.argument('kwargs')
def main(command, kwargs):

    kwargs = json.loads(kwargs)

    if command in globals():
        globals()[command].run(**kwargs)
    else:
        raise ValueError('command not recognized: "{}"'.format(command))

if __name__ == '__main__':
    main()
