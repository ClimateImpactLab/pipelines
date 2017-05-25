'''
Register pipeline segments to be performed on data during a run

.. note::

    Globals will NOT be preserved in these transformations. See the
    :py:mod:`dill` docs for more info.
'''

from __future__ import absolute_import
import pipelines

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
def tasmax_over_27C(ds):
    '''
    Count of days with tasmax gte 27C
    '''

    ds['gte_27'] = ds.tasmax.where((ds.tasmax -273.15) >= 27.).count(dim='time')
    ds['gt_0_lte_27'] = ds.tasmax.where(0 < ds.tasmax - 273.15 < 27).count(dim='time')

    return ds

@pipelines.prep_func
def average_seasonal_temp_pattern(ds):
    '''
    Average seasonal tas
    '''
    return ds.tas.mean(dim='day')
