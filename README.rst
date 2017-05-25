=========================================
Climate Impact Lab Data Pipelines
=========================================

.. image:: https://travis-ci.org/ClimateImpactLab/pipelines.svg?branch=master
        :target: https://travis-ci.org/ClimateImpactLab/pipelines?branch=master

.. image:: https://coveralls.io/repos/github/ClimateImpactLab/pipelines/badge.svg?branch=master
        :target: https://coveralls.io/github/ClimateImpactLab/pipelines?branch=master

.. image:: https://readthedocs.org/projects/pipelines/badge/?version=latest
        :target: https://pipelines.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/ClimateImpactLab/pipelines/shield.svg
        :target: https://pyup.io/repos/github/ClimateImpactLab/pipelines/
        :alt: Updates


The ``pipelines`` package is the place to request, test, and submit jobs at the
Climate Impact Lab. For help submitting a job, see our 
`docs <https://pipelines.readthedocs.io/en/latest/>`_ or ask Justin.

Features
--------

* Tested reshape operations on climate data
* A template for more portions of our pipeline


Usage
-----

  1.  Create a new branch for your request (``git branch my-new-run``)

  2.  In the relevant sector's submodule (e.g. ``pipelines/climate``), look for a template job, e.g. ``/pipelines/climate/jobs/job_bcsd_template.py``

  2.  Copy this template to a directory for your project (e.g. ``gcp-labor`` or ``impactlab_website``)

  3.  Make sure this folder has an ``__init__.py`` file in it. It can be blank.

  4.  Modify the template to your needs. If you need a new transformation, create one in ``transformations.py``

  5.  Initialize your pipelines by importing your file, e.g.: 

      .. code-block:: python
    
          python -m pipelines.climate.jobs.impactlab_website.my_new_job

  6.  Run tests: ``pytest``

  7.  Push your changes to github and file a pull request


Requirements
------------

For now, pipelines requires python 2.7. We're working on 3x support.


Todo
----

See `issues <https://github.com/ClimateImpactLab/pipelines/issues>`_ to see and add to our todos.


Credits
---------

This package was created by `Justin Simcock <https://github.com/jgerardsimcock>`_ and `Michael Delgado <https://github.com/delgadom>`_ of the `Climate Impact Lab <http://impactlab.org>`_. Check us out on `github <https://github.com/ClimateImpactLab>`_.
