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
    pattern_transform)


__author__ = 'Justin Simcock'
__contact__ = 'jsimcock@rhg.com'
__version__ = '0.1.0'

BASELINE_FILE = (
    '/global/scratch/jiacany/nasa_bcsd/pattern/baseline/' +
    '{baseline_model}/{variable}/' +
    '{variable}_baseline_1986-2005_r1i1p1_{baseline_model}_{season}.nc')

BCSD_pattern_files = (
    '/global/scratch/jiacany/nasa_bcsd/pattern/SMME_surrogate/' +
    '{rcp}/{variable}/{model}/' +
    '{variable}_BCSD_{model}_{rcp}_r1i1p1_{season}_{{year}}.nc')

WRITE_PATH = (
    '/global/scratch/mdeglado/web/gcp/climate/{rcp}/{agglev}/{variable}/' +
    '{variable}_{agglev}_{aggwt}_{model}_{season}_{pername}.nc')

ADDITIONAL_METADATA = dict(
    description=__file__.__doc__,
    author=__author__,
    contact=__contact__,
    version=__version__,
    repo='https://github.com/ClimateImpactLab/pipelines',
    file='/pipelines/climate/jobs/impactlab_website/job_pattern_bcsd_ir_slurm.py',
    execute='job_pattern_bcsd_ir_slurm.job_pattern_bcsd_ir_slurm.run_slurm()',
    project='gcp', 
    team='climate',
    geography='hierid',
    weighting='areawt',
    frequency='20yr')

JOBS = [
    dict(variable='tas', transformation=trn.average_seasonal_temp_pattern)]

PERIODS = [
    dict(rcp='rcp85', pername='dummy', years=list(range(2030, 2031)))]
    #dict(rcp='historical', pername='1986', years=list(range(1986, 2006))),
    # dict(rcp='rcp85', pername='2020', years=list(range(2020, 2040))),
    # dict(rcp='rcp85', pername='2040', years=list(range(2040, 2060))),
    # dict(rcp='rcp85', pername='2080', years=list(range(2080, 2100)))]

MODELS = list(map(lambda x: dict(model=x[0], baseline_model=x[1]), [
        # ('pattern1','MRI-CGCM3'),
        # ('pattern2','GFDL-ESM2G'),
        # ('pattern3','MRI-CGCM3'),
        # ('pattern4','GFDL-ESM2G'),
        # ('pattern5','MRI-CGCM3'),
        # ('pattern6','GFDL-ESM2G'),
        # ('pattern28','GFDL-CM3'),
        # ('pattern29','CanESM2'),
        # ('pattern30','GFDL-CM3'),
        # ('pattern31','CanESM2'), 
        # ('pattern32','GFDL-CM3'), 
        ('pattern33','CanESM2')]))

SEASONS = list(map(lambda x: dict(season=x),[ 'DJF']))
# SEASONS = list(map(lambda x: dict(season=x),[ 'DJF', 'MAM', 'JJA', 'SON']))

AGGREGATIONS = [
    # {'agglev': 'ISO', 'aggwt': 'areawt'},
    {'agglev': 'hierid', 'aggwt': 'areawt'}]


@pipelines.register('job_pattern_bcsd_ir_slurm')
@pipelines.add_metadata(ADDITIONAL_METADATA)
@pipelines.read_patterns(
    pattern_file=BCSD_pattern_files,
    baseline_file=BASELINE_FILE)
@pipelines.write_pattern(WRITE_PATH)
@pipelines.iterate(JOBS, PERIODS, MODELS, SEASONS, AGGREGATIONS)
@pipelines.run(workers=1)
def job_pattern_bcsd_ir_slurm():
    return pattern_transform
