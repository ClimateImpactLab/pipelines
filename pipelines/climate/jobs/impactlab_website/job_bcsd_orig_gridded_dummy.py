'''
Dummy data for use in web development

This data is meant to be used for example purposes only. While the intention
is that this data be representative of the variables presented, it is not final
and should not be used in production.
'''

from __future__ import absolute_import
import os

import pipelines
import pipelines.climate.transformations as trn

from pipelines.climate.toolbox import (
    load_climate_data,
    weighted_aggregate_grid_to_regions,
    bcsd_transform)


__author__ = 'Michael Delgado'
__contact__ = 'mdelgado@rhg.com'
__version__ = '0.0.1a1'

BCSD_orig_files = (
    '/shares/gcp/sources/BCSD-original/{rcp}/day/atmos/{variable}/r1i1p1/' +
    'v1.0/{variable}_day_BCSD_{rcp}_r1i1p1_{model}_{{year}}.nc')

WRITE_PATH = os.path.join(
    '/shares/gcp/outputs/diagnostics/web/gcp/climate',
    '{variable}/{variable}_{agglev}_{model}_{pername}.nc')

ADDITIONAL_METADATA = dict(
    description=__doc__.strip(),
    author=__author__,
    contact=__contact__,
    version=__version__,
    project='gcp', 
    team='climate',
    geography='hierid',
    weighting='areawt',
    frequency='year_sample')

JOBS = [
    dict(variable='tasmax', transformation=trn.tasmax_over_95F),
    dict(variable='tasmin', transformation=trn.tasmin_under_32F),
    dict(variable='tas', transformation=trn.average_seasonal_temp)]

PERIODS = [
    dict(rcp='historical', pername='1986', years=[1996]),
    # dict(rcp='rcp85', pername='2020-2039', years=[2030]),
    # dict(rcp='rcp85', pername='2040-2059', years=[2050]),
    dict(rcp='rcp85', pername='2080-2099', years=[2090])]

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

AGGREGATIONS = [{'agglev': 'grid025', 'aggwt': 'unweighted'}]

@pipelines.register('bcsd_orig_gridded_dummy')
@pipelines.add_metadata(ADDITIONAL_METADATA)
@pipelines.read_patterns(BCSD_orig_files)
@pipelines.write_pattern(WRITE_PATH)
@pipelines.iterate(JOBS, PERIODS, MODELS, AGGREGATIONS)
@pipelines.run(workers=1)
def bcsd_orig_gridded_dummy(*args, **kwargs):
    return bcsd_transform
