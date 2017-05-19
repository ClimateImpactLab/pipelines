'''
Processed BCSD-orig 20-year average IR-level data by model

Aggregated to impact regions using area weights. Produced from BCSD-originals
using the GCP climate toolbox at https://github.com/ClimateImpactLab/pipelines
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
__version__ = '0.1.0'

BCSD_orig_files = os.path.join(
    '/shares/gcp/sources/BCSD-original/{rcp}/day/atmos/{variable}/r1i1p1/v1.0',
    '{variable}_day_BCSD_{rcp}_r1i1p1_{model}_{year}.nc')

WRITE_PATH = os.path.join(
    '/shares/gcp/outputs/diagnostics/web/gcp/climate',
    '{variable}/{variable}_{model}_{period}.nc')

ADDITIONAL_METADATA = dict(
    description=__file__.__doc__,
    author=__author__,
    contact=__contact__,
    version=__version__,
    repo='https://github.com/ClimateImpactLab/pipelines',
    file='/climate/jobs/impactlab-website/bcsd-orig.py',
    execute='climate.jobs.impactlab_website.bcsd_orig.main',
    project='gcp', 
    team='climate',
    geography='hierid',
    weighting='areawt',
    frequency='20yr')


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
    return ds.tas.groupby('time.season').mean(dim='time')

JOBS = [
    dict(variable='tasmax', transformation=tasmax_over_95F),
    dict(variable='tasmin', transformation=tasmin_under_32F),
    dict(variable='tas', transformation=average_seasonal_temp)]

PERIODS = [
    dict(rcp='historical', pername='1986', years=list(range(1986, 2006))),
    dict(rcp='rcp85', pername='2020', years=list(range(2020, 2040))),
    dict(rcp='rcp85', pername='2040', years=list(range(2040, 2060))),
    dict(rcp='rcp85', pername='2080', years=list(range(2080, 2100)))]

MODELS = list(map(lambda x: dict(model=x), [
    'ACCESS1-0',
    'bcc-csm1-1',
    'BNU-ESM',
    'CanESM2',
    'CCSM4',
    'CESM1-BGC',
    'CNRM-CM5',
    'CSIRO-Mk3-6-0',
    'GFDL-CM3',
    'GFDL-ESM2G',
    'GFDL-ESM2M',
    'IPSL-CM5A-LR',
    'IPSL-CM5A-MR',
    'MIROC-ESM-CHEM',
    'MIROC-ESM',
    'MIROC5',
    'MPI-ESM-LR',
    'MPI-ESM-MR',
    'MRI-CGCM3',
    'inmcm4',
    'NorESM1-M']))

ITERATION_COMPONENTS = (JOBS, PERIODS, MODELS)


def run_job(variable, transformation, rcp, pername, years, model):

    # Build job metadata
    metadata = {k: v for k, v in ADDITIONAL_METADATA.items()}
    metadata.update(dict(
        variable=variable, model=model, period=pername, rcp=rcp))

    metadata['transformation'] = transformation.__doc__
    metadata['time_horizon'] = '{}-{}'.format(years[0], years[-1])

    # Get transformed data
    transformed = xr.concat([
        (load_climate_data(
                    BCSD_orig_files.format(year=y, **metadata),
                    variable)
            .pipe(transformation))
        for y in years],
        dim=pd.Index(years, name='year')).mean(dim='year')
    
    # Reshape to regions
    wtd = weighted_aggregate_grid_to_regions(
            transformed, variable, 'areawt', 'hierid')

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
