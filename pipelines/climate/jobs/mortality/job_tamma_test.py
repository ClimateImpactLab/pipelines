'''
Test to profile transformations on daily climate values
'''


from __future__ import absolute_import
import os

import time
import pipelines
import pipelines.climate.transformations as trn

from pipelines.climate.toolbox import (
    load_climate_data,
    weighted_aggregate_grid_to_regions,
    bcsd_transform_annual)


__author__ = 'Justin Simcock'
__contact__ = 'jsimcock@rhg.com'
__version__ = '0.1.0'

BCSD_orig_files  = (
    '/global/scratch/jiacany/nasa_bcsd/raw_data/{rcp}/' +
    '{model}/{variable}/' +
    '{variable}_day_BCSD_{rcp}_r1i1p1_{model}_{{year}}.nc')


WRITE_PATH = os.path.join(
    '/global/scratch/jsimcock/gcp/climate/{agglev}/{rcp}',
    '{variable}/{transformation_name}/{model}',
    '{variable}_{transformation_name}_{model}_{{year}}.nc')



ADDITIONAL_METADATA = dict(
    description=__doc__.strip(),
    author=__author__,
    contact=__contact__,
    version=__version__,
    repo='https://github.com/ClimateImpactLab/pipelines',
    file='/climate/jobs/mortality/job_tamma_test.py',
    execute='climate.jobs.mortality.job_tamma_test.main',
    project='gcp', 
    team='climate',
    geography='hierid',
    weighting='popwt',
    frequency='daily')

JOBS = [
    dict(variable='tas', transformation_name='tas-polynomials', transformation=trn.polynomials) 

    ]

PERIODS = [#dict(rcp='historical' , pername='annual', years=list(range(1981, 2006)))]
            dict(rcp='rcp85', pername='annual', years=list(range(2006, 2100)))]

MODELS = list(map(lambda x: dict(model=x), [
    # 'ACCESS1-0',
    'bcc-csm1-1',
    # 'BNU-ESM',
    'CanESM2',
    # 'CCSM4',
    'CESM1-BGC',
    'CNRM-CM5',
    #'CSIRO-Mk3-6-0',
    'GFDL-CM3',
    'GFDL-ESM2G',
    'GFDL-ESM2M',
    'IPSL-CM5A-LR',
    'IPSL-CM5A-MR',
    'MIROC-ESM-CHEM',
    # 'MIROC-ESM',
    'MIROC5',
    'MPI-ESM-LR',
    'MPI-ESM-MR',
    'MRI-CGCM3',
    'inmcm4',
    'NorESM1-M']))

AGGREGATIONS = [{'agglev': 'hierid', 'aggwt': 'popwt'}]


@pipelines.register('job_tamma_test')
@pipelines.add_metadata(ADDITIONAL_METADATA)
@pipelines.read_patterns(BCSD_orig_files)
@pipelines.write_pattern(WRITE_PATH)
@pipelines.iterate(JOBS, PERIODS, MODELS, AGGREGATIONS)
@pipelines.run(workers=1)
def job_tamma_test(*args, **kwargs):
    return bcsd_transform_annual