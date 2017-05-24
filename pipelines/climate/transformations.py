
from __future__ import absolute_import
import pipelines

@pipelines.document
def tasmin_under_32F(ds):
    '''
    Count of days with tasmin under 32F/0C
    '''
    return ds.tasmin.where((ds.tasmin- 273.15) < 0).count(dim='time')


@pipelines.document
def tasmax_over_95F(ds):
    '''
    Count of days with tasmax over 95F/35C
    '''
    return ds.tasmax.where((ds.tasmax- 273.15) > 35).count(dim='time')


@pipelines.document
def average_seasonal_temp(ds):
    '''
    Average seasonal tas
    '''
    return ds.tas.groupby('time.season').mean(dim='time')


# @pipelines.document
# def example_transformation_annual_binned_tas(ds):
#     '''
#     Example computation: annual binned tas
    
#     See also
#     --------
#     xarray.DataArray.groupby_bins
#     pandas.cut

#     '''

#     bins_tas_C = [0, 10, 14, 16, 19, 23, 31, 40]


#     return (ds
#         .groupby_bins(
#               'tas',  # name of the variable to be grouped
#               bins_tas_C,  # bin definitions
#               right=True,  # right-inclusive, i.e. (0, 10], (10, 14], ...
#               include_lowest=True  # first interval should be left-inclusive
#               )
#         .count(dim='time'))  # count number of days in each bin for the year
