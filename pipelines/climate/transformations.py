'''
Register pipeline segments to be performed on data during a run

.. note::

    Globals will NOT be preserved in these transformations. See the
    :py:mod:`dill` docs for more info.
'''

from __future__ import absolute_import
import pipelines

# @pipelines.prep_func
# def tasmin_under_32F(ds):
#     '''
#     Count of days with tasmin under 32F/0C
#     '''
#     return ds.tasmin.where((ds.tasmin- 273.15) < 0).count(dim='time')


# @pipelines.prep_func
# def tasmax_over_95F(ds):
#     '''
#     Count of days with tasmax over 95F/35C
#     '''
#     return ds.tasmax.where((ds.tasmax- 273.15) > 35).count(dim='time')


# @pipelines.prep_func
# def average_seasonal_temp(ds):
#     '''
#     Average seasonal tas
#     '''
#     return ds.tas.groupby('time.season').mean(dim='time')


@pipelines.prep_func
def tasmax_over_27C_pow1(ds):
    '''
    Sum of days with tasmax gt 27C
    Sum of tasmax for days with tasmax between 0C and 27C

    '''

    ds['gt_27'] = (ds.tasmax - 273.15).where((ds.tasmax -273.15) > 27.).sum(dim='time')
    ds['gt_0_lt_27'] = (ds.tasmax - 273.15).where((0 < (ds.tasmax - 273.15)) & ((ds.tasmax - 273.15) < 27)).sum(dim='time')

    return ds

@pipelines.prep_func
def tasmax_over_27C_pow2(ds):
    '''
    Sum of tasmax**2 for days with tasmax gt 27C
    Sum of tasmax**2 for days with tasmax between 0C and 27C

    '''

    ds['gt_27'] = (((ds.tasmax - 273.15).where((ds.tasmax -273.15) > 27.))**2).sum(dim='time')
    ds['gt_0_lt_27'] = (((ds.tasmax - 273.15).where((0 < (ds.tasmax - 273.15)) & ((ds.tasmax - 273.15) < 27)))**2).sum(dim='time')

    return ds

@pipelines.prep_func
def tasmax_over_27C_pow3(ds):
    '''
    Sum of tasmax**3 for days with tasmax gt 27C
    Sum of tasmax**3 for days with tasmax between 0C and 27C
    '''

    ds['gt_27'] = (((ds.tasmax - 273.15).where((ds.tasmax -273.15) > 27.))**3).sum(dim='time')
    ds['gt_0_lt_27'] = (((ds.tasmax - 273.15).where((0 < (ds.tasmax - 273.15)) & ((ds.tasmax - 273.15) < 27)))**3).sum(dim='time')

    return ds

@pipelines.prep_func
def tasmax_over_27C_pow4(ds):
    '''
    Sum of tasmax**4 for days with tasmax gt 27C
    Sum of tasmax**4 for days with tasmax between 0C and 27C

    ''' 
    ds['gt_27'] = (((ds.tasmax - 273.15).where((ds.tasmax -273.15) > 27.))**4).sum(dim='time')
    ds['gt_0_lt_27'] = (((ds.tasmax - 273.15).where((0 < (ds.tasmax - 273.15)) & ((ds.tasmax - 273.15) < 27)))**4).sum(dim='time')

    return ds


# @pipelines.prep_func
# def average_seasonal_temp_pattern(ds):
#     '''
#     Average seasonal tas
#     '''
#     return (ds.tas - 273.15).mean(dim='day')
