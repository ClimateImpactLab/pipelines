'''
Register pipeline segments to be performed on data during a run

.. note::

    Globals will NOT be preserved in these transformations. See the
    :py:mod:`dill` docs for more info.
'''

from __future__ import absolute_import
import pipelines
import time

@pipelines.prep_func
def tasmin_under_32F(ds):
    '''
    Count of days with tasmin under 32F/0C
    '''
    return ds.tasmin.where((ds.tasmin- 273.15) < 0).count(dim='time')


@pipelines.prep_func
def tasmax_over_95F(ds):
    '''
    Count of days with tasmax over 95F/35C
    '''
    return ds.tasmax.where((ds.tasmax- 273.15) > 35).count(dim='time')


@pipelines.prep_func
def average_seasonal_temp(ds):
    '''
    Average seasonal tas
    '''
    return ds.tas.groupby('time.season').mean(dim='time')


@pipelines.prep_func
def tasmax_over_27C_pow1(ds):
    '''
    Sum of days with tasmax gte 27C
    '''

    ds['gte_27'] = ds.tasmax.where((ds.tasmax -273.15) >= 27.).sum(dim='time')
    ds['gt_0_lte_27'] = ds.tasmax.where((0 < (ds.tasmax - 273.15)) & ((ds.tasmax - 273.15) < 27)).sum(dim='time')

    return ds

@pipelines.prep_func
def tasmax_over_27C_pow2(ds):
    '''
    Sum of tasmax**2 for days with tasmax gte 27C
    '''

    ds['gte_27'] = ds.tasmax.where((ds.tasmax -273.15) >= 27.).sum(dim='time')**2
    ds['gt_0_lte_27'] = ds.tasmax.where((0 < (ds.tasmax - 273.15)) & ((ds.tasmax - 273.15) < 27)).sum(dim='time')**2

    return ds

@pipelines.prep_func
def tasmax_over_27C_pow3(ds):
    '''
    Sum of tasmax**3 for days with tasmax gte 27C
    '''

    ds['gte_27'] = ds.tasmax.where((ds.tasmax -273.15) >= 27.).sum(dim='time')**3
    ds['gt_0_lte_27'] = ds.tasmax.where((0 < (ds.tasmax - 273.15)) & ((ds.tasmax - 273.15) < 27)).sum(dim='time')**3

    return ds

@pipelines.prep_func
def tasmax_over_27C_pow4(ds):
    '''
    Sum of tasmax**4 for days with tasmax gte 27C

    '''

    ds['gte_27'] = ds.tasmax.where((ds.tasmax -273.15) >= 27.).sum(dim='time')**4
    ds['gt_0_lte_27'] = ds.tasmax.where((0 < (ds.tasmax - 273.15)) & ((ds.tasmax - 273.15) < 27)).sum(dim='time')**4

    return ds

@pipelines.prep_func
def polynomials(ds):
    '''
    Raises all data variables to all values in ``powers``
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to compute powers from. All data_vars will be
        raised
    powers: 
    '''
    t1 = time.time()

    keys = ds.data_vars.keys()

    for power in range(5):
        if power < 2:
            continue

        for var in keys:
            ds[var + '_{}'.format(power)] = (ds[var] - 273.15)**power

    t2 = time.time()
    print('Polynomial transformation complete: {}'.format(t2-t1))
    return ds




@pipelines.prep_func
def average_seasonal_temp_pattern(ds):
    '''
    Average seasonal tas
    '''
    return (ds.tas - 273.15).mean(dim='day')
