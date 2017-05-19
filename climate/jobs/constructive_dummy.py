'''
Dummy data for use in web development

This data is meant to be used for example purposes only. While the intention
is that this data be representative of the variables presented, it is not final
and should not be used in production.
'''

import os
import itertools

__author__ = 'Michael Delgado'
__contact__ = 'mdelgado@rhg.com'
__version__ = '0.0.1a1'


from toolbox import (
    fill_holes_xr,
    _standardize_longitude_dimension,
    get_period_mean,
    weighted_aggregate_grid_to_regions)


BCSD_orig_files = os.path.join(
    '/shares/gcp/sources/BCSD-original/rcp85/day/atmos/{variable}/r1i1p1/v1.0',
    '{variable}_day_BCSD_rcp85_r1i1p1_{model}_{period}.nc')

WRITE_PATH = os.path.join(
    '/global/scratch/jsimcock/climate_data/',
    '{variable}/{variable}_{model}_{period}.nc')

ADDITIONAL_METADATA = dict(
    description=__file__.__doc__,
    author=__author__,
    contact=__contact__,
    version=__version__,
    project='gcp', 
    team='climate',
    geography='hierid',
    weighting='areawt',
    frequency='year_sample')


def tasmin_under_32F(ds):
    '''
    Count of days with tasmin under 32F/0C
    '''
    return ds.tasmin.where((ds.tasmin- 273.15) < 0).count(dim='time')


def tasmax_over_95F(ds):
    '''
    Count of days with tasmax over 95F/35C
    '''
    return ds.tasmin.where((ds.tasmin- 273.15) > 35).count(dim='time')


def average_seasonal_temp(ds):
    '''
    Average seasonal tas
    '''
    return ds.tas.groupby({'time': ds['time.season']}).mean(dim='time')


def run_job(var, transformation, pername, years, model):

    # Build job metadata
    metadata = {k: v for k, v in ADDITIONAL_METADATA.items()}
    metadata.update(dict(variable=var, model=model, period=pername))
    metadata['transformation'] = transformation.__doc__
    metadata['time_horizon'] = '{}-{}'.format(years[0], years[-1])

    # Get transformed data
    transformed = get_period_mean(
        BCSD_orig_files.format(**metadata),
        model,
        years,
        transform_climate_data)

    # Reshape to regions
    wtd = weighted_aggregate_grid_to_regions(
            transformed, weights_fp, 'areawt', 'hierid')

    # Update netCDF metadata
    wtd.attrs.update(**metadata)

    # Write output
    wtd.to_netcdf(WRITE_PATH.format(**metadata))


def main():

    jobs = [
        ('tasmax', tasmax_over_95F),
        ('tasmin', tasmin_under_32F),
        ('tas', average_seasonal_temp)]

    periods = [
        ('2020', [2030]),
        ('2040', [2050]),
        ('2080', [2090])]

    models = ['ACCESS1-0','CESM1-BGC', 'GFDL-ESM2M']

    for job, period, model in itertools.product(jobs, periods, models):

        var, transformation = job
        pername, years = period

        run_job(var, transformation, pername, years, model)


if __name__ == '__main__':
    main()
