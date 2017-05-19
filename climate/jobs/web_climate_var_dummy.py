'''
Dummy data for use in web development

This data is meant to be used for example purposes only. While the intention
is that this data be representative of the variables presented, it is not final
and should not be used in production.
'''

from __future__ import absolute_import
import os
import itertools
import logging
from functools import reduce
import xarray as xr
import pandas as pd

from climate.toolbox import (
    load_climate_data,
    weighted_aggregate_grid_to_regions)


__author__ = 'Michael Delgado'
__contact__ = 'mdelgado@rhg.com'
__version__ = '0.0.1a1'

BCSD_orig_files = os.path.join(
    '/shares/gcp/sources/BCSD-original/rcp85/day/atmos/{variable}/r1i1p1/v1.0',
    '{variable}_day_BCSD_rcp85_r1i1p1_{model}_{year}.nc')

WRITE_PATH = os.path.join(
    '/shares/gcp/outputs/diagnostics/web/gcp/climate',
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
    return ds.tasmax.where((ds.tasmax- 273.15) > 35).count(dim='time')


def average_seasonal_temp(ds):
    '''
    Average seasonal tas
    '''
    return ds.tas.groupby({'time': ds['time.season']}).mean(dim='time')

JOBS = [
    dict(variable='tasmax', transformation=tasmax_over_95F),
    dict(variable='tasmin', transformation=tasmin_under_32F),
    dict(variable='tas', transformation=average_seasonal_temp)]

PERIODS = [
    dict(pername='2020', years=[2030]),
    dict(pername='2040', years=[2050]),
    dict(pername='2080', years=[2090])]

MODELS = [
    dict(model='ACCESS1-0'),
    dict(model='CESM1-BGC'),
    dict(model='GFDL-ESM2M')]

ITERATION_COMPONENTS = (JOBS, PERIODS, MODELS)


def run_job(variable, transformation, pername, years, model):

    # Build job metadata
    metadata = {k: v for k, v in ADDITIONAL_METADATA.items()}
    metadata.update(dict(variable=variable, model=model, period=pername))
    metadata['transformation'] = transformation.__doc__
    metadata['time_horizon'] = '{}-{}'.format(years[0], years[-1])

    # Get transformed data
    transformed = xr.concat([
        (load_climate_data(
            BCSD_orig_files.format(year=y, model=model, variable=variable),
                variable)
            .pipe(transformation))
        for y in years],
        dim=pd.Index(years, name='year')).mean(dim='year')
    
    # Reshape to regions
    wtd = weighted_aggregate_grid_to_regions(
            transformed, weights_fp, 'areawt', 'hierid')

    # Update netCDF metadata
    wtd.attrs.update(**metadata)

    # Write output
    fp = WRITE_PATH.format(**metadata)
    if not os.path.isdir(os.path.dirname(fp)):
        os.makedirs(os.path.dirname(fp))
    wtd.to_netcdf(fp)


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('uploader')
logger.setLevel('INFO')


def main():

    njobs = reduce(lambda x, y: x*y, map(len, ITERATION_COMPONENTS))

    for i, job_components in enumerate(
            itertools.product(*ITERATION_COMPONENTS)):

        job = {}
        for job_component in job_components:
            job.update(job_component)

        logger.info('beginning job {} of {}'.format(i, njobs))

        try:
            run_job(**job)
        except Exception, e:
            logger.error(
                'Error encountered in job {} of {}:\n\nJob spec:\n{}\n\n'
                    .format(i, njobs, job),
                exc_info=e)


if __name__ == '__main__':
    main()
