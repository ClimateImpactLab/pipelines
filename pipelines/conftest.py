
import pytest
import inspect
import pipelines
import os

try:
    from importlib import import_module
except ImportError:
    from imp import load_module as import_module


def check_module(path):
    if not str(path).endswith('.py'):
        return False

    excludes = ['__init__.py', 'conftest.py', 'pipelines.py']

    if os.path.basename(str(path)) not in excludes:
        if '__pycache__' in str(path):
            return False

        return True


def pytest_collect_file(parent, path):

    rootpath = path.relto(parent.config.rootdir).replace(os.sep, "/")
    if path.basename.startswith('test') and check_module(path.realpath()):
        pass

    elif check_module(path.realpath()):

        return PipelineModule(path, parent)


class PipelineModule(pytest.Module):
    def collect(self):
        modpath = os.path.splitext(self.name)[0].replace('/', '.')
        mod = import_module(modpath)

        for name, obj in inspect.getmembers(mod):

            if isinstance(obj, pipelines.JobCreator):
                yield BCSDItem(name, self, obj())


class BCSDItem(pytest.Item):
    def __init__(self, name, parent, job):
        super(BCSDItem, self).__init__(name, parent)
        self.job = job

    def runtest(self):
        # do the test
        self.job.test()

