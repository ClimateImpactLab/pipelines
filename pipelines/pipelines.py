
import itertools
import xarray as xr
import pandas as pd
import numpy as np

import shutil
import os
import tempfile

from contextlib import contextmanager


@contextmanager
def temporary_dir():
    d = tempfile.mkdtemp()

    try:
        yield d

    finally:
        shutil.rmtree(d)


class JobRunner(object):

    def __init__(
            self,
            name,
            func,
            iteration_components,
            read_pattern,
            write_pattern,
            workers=1,
            metadata={}):

        self._name = name
        self._func = func
        self._iteration_components = iteration_components
        self._read_pattern = read_pattern
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

    def _run_one_job(self, job, read_pattern, write_pattern):
        metadata = {k: v for k, v in self._metadata.items()}
        metadata.update({k: str(v) for k, v in job.items()})
        self._func(read_pattern, write_pattern, metadata=metadata, **job)

    def run(self):
        for job in self._get_jobs():
            _run_one_job(job, self._read_pattern, self._write_pattern)

    def test(self):

        i = None

        with temporary_dir() as tmp:
        
            for i, job in enumerate(self._get_jobs()):
                assert len(job) > 0, 'No job specification in job {}'.format(i)

                # ideally, test to make sure all the inputs exist on datafs
                # check_datafs(job)
            
                variable = job.get('variable', 'tas')

                tmp_path_in = os.path.join(tmp, 'sample_in.nc')

                ds = xr.Dataset({
                    variable: xr.DataArray(
                        np.random.random((365, 37, 73)),
                        dims=('time', 'lat', 'lon'),
                        coords={
                            'time': pd.date_range('1/1/1981', periods=365, freq='D'),
                            'lat': np.linspace(-90, 90, 37),
                            'lon': np.linspace(0, 360, 73)})
                    })

                ds.to_netcdf(tmp_path_in)

                tmp_path_out = os.path.join(tmp, 'sample_out.nc')

                # Check functions with last job
                res = self._run_one_job(job, tmp_path_in, tmp_path_out)

                assert os.path.isfile(tmp_path_out), "No file created"
                os.remove(tmp_path_out)

        if i is None:
            raise ValueError('No jobs specified')


class JobCreator(object):
    def __init__(self, name, func):
        self._name = name
        self._func = func

    def __call__(self, *args, **kwargs):
        kwargs.update({'name': self._name})
        return self._func(*args, **kwargs)


def register(name):
    def decorator(func):
        return JobCreator(name, func)
    return decorator


def run(workers=1):
    def decorator(func):
        def inner(*args, **kwargs):
            kwargs.update(dict(func=func, workers=workers))
            return JobRunner(*args, **kwargs)
        return inner
    return decorator


def read_pattern(patt):
    def decorator(func):
        def inner(*args, **kwargs):
            kwargs.update(dict(read_pattern=patt))
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


def iter(*iters):
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
