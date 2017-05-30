'''
Example BCSD transformation pipeline definition file
'''


from __future__ import absolute_import
import os

import pipelines
import pipelines.climate.transformations as trn

from pipelines.climate.toolbox import (
    load_climate_data,
    weighted_aggregate_grid_to_regions,
    bcsd_transform_annual)


__author__ = 'Justin Gerard'
__contact__ = 'jsimcock@rhg.com'
__version__ = '0.1.0'

# BCSD_orig_files  = (
#     '/global/scratch/jiacany/nasa_bcsd/raw_data/{rcp}/' +
#     '{model}/{variable}/' +
#     '{variable}_day_BCSD_{rcp}_r1i1p1_{model}_{{year}}.nc')

##sacag test

BCSD_orig_files = os.path.join(
    '/shares/gcp/sources/BCSD-original/{rcp}/day/atmos/{variable}/r1i1p1/v1.0',
    '{variable}_day_BCSD_{rcp}_r1i1p1_{model}_{{year}}.nc')

WRITE_PATH = os.path.join(
    '~/data/{agglev}/{rcp}',
    '{variable}/{transformation_name}/{model}',
    '{variable}_{transformation_name}_{model}_{{year}}.nc')

# WRITE_PATH = os.path.join(
#     '/global/scratch/jsimcock/gcp/climate/{agglev}/{rcp}',
#     '{variable}/{transformation_name}/{model}',
#     '{variable}_{transformation_name}_{model}_{{year}}.nc')




ADDITIONAL_METADATA = dict(
    description=__doc__.strip(),
    author=__author__,
    contact=__contact__,
    version=__version__,
    repo='https://github.com/ClimateImpactLab/pipelines',
    file='/climate/jobs/impactlab-website/job_bcsd_ir_slurm_labor.py',
    execute='climate.jobs.impactlab_website.job_bcsd_ir_slurm_labor.main',
    project='gcp', 
    team='climate',
    geography='hierid',
    weighting='popwt',
    frequency='annual')

JOBS = [
    dict(variable='tasmax', transformation_name='tasmax-over-27C-pow1', transformation=trn.tasmax_over_27C_pow1),
    dict(variable='tasmax', transformation_name='tasmax-over-27C-pow2', transformation=trn.tasmax_over_27C_pow2), 
    dict(variable='tasmax', transformation_name='tasmax-over-27C-pow3', transformation=trn.tasmax_over_27C_pow3), 
    dict(variable='tasmax', transformation_name='tasmax-over-27C-pow4', transformation=trn.tasmax_over_27C_pow4), 

    ]

PERIODS = [dict(rcp='historical' , pername='annual', years=list(range(1981, 1986)))]
        # dict(rcp='rcp85', pername='annual', years=list(range(2006, 2100))))

MODELS = list(map(lambda x: dict(model=x), [
    'ACCESS1-0']))
    # 'bcc-csm1-1',
    # 'BNU-ESM',
    # 'CanESM2',
    # 'CCSM4',
    # 'CESM1-BGC',
    # 'CNRM-CM5',
    # 'CSIRO-Mk3-6-0',
    # 'GFDL-CM3',
    # 'GFDL-ESM2G',
    # 'GFDL-ESM2M',
    # 'IPSL-CM5A-LR',
    # 'IPSL-CM5A-MR',
    # 'MIROC-ESM-CHEM',
    # 'MIROC-ESM',
    # 'MIROC5',
    # 'MPI-ESM-LR',
    # 'MPI-ESM-MR',
    # 'MRI-CGCM3',
    # 'inmcm4',
    # 'NorESM1-M']))

AGGREGATIONS = [{'agglev': 'hierid', 'aggwt': 'popwt'}]


@pipelines.register('job_bcsd_ir_slurm_labor')
@pipelines.add_metadata(ADDITIONAL_METADATA)
@pipelines.read_patterns(BCSD_orig_files)
@pipelines.write_pattern(WRITE_PATH)
@pipelines.iterate(JOBS, PERIODS, MODELS, AGGREGATIONS)
@pipelines.run(workers=1)
def job_bcsd_ir_slurm_labor(*args, **kwargs):
    return bcsd_transform_annual
