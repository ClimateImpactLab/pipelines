
import itertools
import xarray as xr
import pandas as pd
import numpy as np

import shutil
import os
import tempfile
import dill
import json
import inspect

from toolz import memoize
from contextlib import contextmanager

import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('uploader')
logger.setLevel('INFO')


@contextmanager
def temporary_dir():
    d = tempfile.mkdtemp()

    try:
        yield d

    finally:
        shutil.rmtree(d)


@memoize
def create_dummy_data(tmp, variable):

    tmp_path_in = os.path.join(tmp, 'sample_in.nc')

    time = pd.date_range('1/1/1981', periods=4, freq='3M')
    lats = np.arange(-89.875, 90, 0.25)
    lons = np.arange(-179.875, 180, 0.25)

    ds = xr.Dataset({
        variable: xr.DataArray(
            np.random.random((len(time), len(lats), len(lons))),
            dims=('time', 'lat', 'lon'),
            coords={
                'time': time,
                'lat': lats,
                'lon': lons})
        })

    ds.to_netcdf(tmp_path_in)

    return tmp_path_in


class JobRunner(object):
    '''
    Generalized job dispatch class
    '''

    def __init__(
            self,
            name,
            func,
            iteration_components,
            read_patterns,
            write_pattern,
            workers=1,
            metadata={}):

        self._name = name
        self._job_function_getter = func
        self._iteration_components = iteration_components
        self._read_patterns = read_patterns
        self._write_pattern = write_pattern
        self._njobs = reduce(
            lambda x, y: x*y, map(len, self._iteration_components))
        self._metadata = metadata

    def _get_jobs(self):

        for i, job_components in enumerate(
                itertools.product(*self._iteration_components)):

            job = {}
            for job_component in job_components:
                job.update(job_component)

            yield job

    def _build_metadata(self, job):
        metadata = {k: v for k, v in self._metadata.items()}
        metadata.update({k: str(v) for k, v in job.items()})

        return metadata

    def _run_one_job(self, job):

        func = self._job_function_getter()
        try:
            func(**job)
        except TypeError:
            print(job)
            raise

    def run(self):
        '''
        Invoke a full run for the specified job set
        '''

        for i, job in enumerate(self._get_jobs()):
            logger.info('beginning job {} of {}'.format(i, self._njobs))
            try:
                metadata = self._build_metadata(job)
                    
                kwargs = {k: v for k, v in job.items()}
                kwargs.update(
                    {k: v.format(**metadata) for k, v in self._read_patterns.items()})
                kwargs['write_file'] = self._write_pattern.format(**metadata)
                kwargs['metadata'] = metadata

                self._run_one_job(kwargs)

            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception, e:
                logging.error(
                    'Error encountered in job {} of {}:\nJob spec:\n{}\n'
                        .format(i, self._njobs, job),
                    exc_info=e)

    def run_slurm(self):

        func = self._job_function_getter()

        for i, job in enumerate(self._get_jobs()):

            run_flags = [
                '--job-name={}_{}'.format(self._name, i),
                '--output={}_{}'.format(self.name, i),
                '--partition=savio2_bigmem',
                '--account=co_laika',
                '--qos=laika_bigmem2_normal',
                '--nodes=1',
                '--ntasks-per-node=20',
                '--cpus-per-task=1',
                '--time=72:00:00']

            metadata = self._build_metadata(job)
            
            kwargs = {k: v for k, v in job.items()}
            kwargs.update(
                {k: v.format(**metadata)
                    for k, v in self._read_patterns.items()})
            kwargs['write_file'] = self._write_pattern.format(**metadata)
            kwargs['metadata'] = metadata

            # logger.info('beginning job {} of {}'.format(i, self._njobs))
            os.system("srun {flags} run python -m {module} {func} '{job}'".format(
                flags=' '.join(run_flags),
                module=func.__module__,
                func=func.__name__,
                job=json.dumps(kwargs)))


    def test(self):
        '''
        Test the specified run using dummy data
        '''

        i = None

        with temporary_dir() as tmp:
        
            for i, job in enumerate(self._get_jobs()):
                assert len(job) > 0, 'No job specification in job {}'.format(i)

                # Ensure paths are specified correctly
                # Don't check for presence, but check pattern
                for pname, patt in self._read_patterns.items():
                    assert len(patt.format(**job)) > 0

                assert len(self._write_pattern.format(**job)) > 0

                # ideally, test to make sure all the inputs exist on datafs
                # check_datafs(job)
            
                variable = job.get('variable', 'tas')
                kwargs = {k: v for k, v in job.items()}

                dummy = create_dummy_data(tmp, variable)
                kwargs.update({k: dummy for k in self._read_patterns.keys()})
                kwargs['write_file'] = os.path.join(tmp, 'sample_out.nc')
                kwargs['metadata'] = self._build_metadata(job)

                # Check functions with last job
                res = self._run_one_job(kwargs)

                assert os.path.isfile(tmp_path_out), "No file created"
                os.remove(tmp_path_out)

        if i is None:
            raise ValueError('No jobs specified')


class JobCreator(object):
    def __init__(self, name, func):
        self._name = name
        self._job_function_getter = func

    def __call__(self, *args, **kwargs):
        kwargs.update({'name': self._name})
        return self._job_function_getter(*args, **kwargs)


def register(name):
    def decorator(func):
        return JobCreator(name, func)
    return decorator


def read_patterns(*patt, **patterns):
    if len(patt) == 1 and 'read_pattern' not in patterns:
        patterns['read_file'] = patt[0]
    elif len(patt) > 1:
        raise ValueError('more than one read pattern must use kwargs')

    def decorator(func):
        def inner(*args, **kwargs):
            kwargs.update({'read_patterns': patterns})
            return func(*args, **kwargs)
        return inner
    return decorator


def write_pattern(patt):
    def decorator(func):
        def inner(*args, **kwargs):
            kwargs.update(dict(write_pattern=patt))
            return func(*args, **kwargs)
        return inner
    return decorator


def iterate(*iters):
    def decorator(func):
        def inner(*args, **kwargs):
            kwargs.update(dict(iteration_components=iters))
            return func(*args, **kwargs)
        return inner
    return decorator


def add_metadata(metadata):
    def decorator(func):
        def inner(*args, **kwargs):
            kwargs.update(dict(metadata=metadata))
            return func(*args, **kwargs)
        return inner
    return decorator


def run(workers=1):
    def decorator(func):
        def inner(*args, **kwargs):
            kwargs.update(dict(func=func, workers=workers))
            return JobRunner(*args, **kwargs)
        return inner
    return decorator


def prep_func(func):

    funcname = '.'.join([inspect.getmodule(func).__name__, func.__name__])

    if not os.path.isdir('pipes'):
        os.makedirs('pipes')

    fp = 'pipes/{}'.format(funcname)
    
    with open(fp, 'wb+') as f:
        pickled = dill.dump(func, f)

    return funcname


def load_func(func):
    fp = 'pipes/{}'.format(func)
    with open(fp, 'rb') as f:
        return dill.load(f)
