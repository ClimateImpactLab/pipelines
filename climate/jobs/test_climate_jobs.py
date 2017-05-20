import pkgutil


def test_submodules():
    
    __all__ = []
    
    for loader, module_name, is_pkg in  pkgutil.walk_packages(__file__):
        if module_name.split('.')[-1].startswith('bcsd'):
            __all__.append(module_name)
            module = loader.find_module(module_name).load_module(module_name)
            assert hasattr(module, 'run_job')
