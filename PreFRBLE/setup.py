# -*- coding: utf-8 -*-

"""
PreFRBLE Setup
==========
Contains the setup script required for installing the *PreFRBLE* package.
This can be ran directly by using::

    pip install .

or anything equivalent.

"""


# %% IMPORTS
# Future imports (only required if py2/py3 compatible)
from __future__ import absolute_import, with_statement

# Built-in imports
from codecs import open

# Package imports
from setuptools import find_packages, setup


# %% SETUP DEFINITION
# Get the long description from the README file
with open('README.rst', 'r') as f:
    long_description = f.read()

# Get the requirements list by reading the file and splitting it up
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Get the version from the __version__.py file
# This is done in this way to make sure it is stored in a single place and
# does not require the package to be installed already.
version = None
with open('PreFRBLE/__version__.py', 'r') as f:
    exec(f.read())

# Setup function declaration
# See https://setuptools.readthedocs.io/en/latest/setuptools.html
setup(name='PreFRBLE',    # Distribution name of package (e.g., used on PyPI)
      version=version,      # Version of this package (see PEP 440)
      author="Stefan Hackstein",
      author_email="stefan.hackstein@hs.uni-hamburg.de",
      maintainer="",   # PyPI username of maintainer(s)
      description=("PreFRBLE: Predict Fast Radio Bursts to obtain model Likelihood Estimates"),
      long_description=long_description,        # Use the README description
      url="https://github.com/shackste/PreFRBLE",
      license='personal',    # License of this package
      # List of classifiers (https://pypi.org/pypi?%3Aaction=list_classifiers)
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Software Development'
          ],
      keywords=('PreFRBLE', 'Fast Radio Bursts', 'Astrophysical Processes'),  # List of keywords
      # String containing the Python version requirements (see PEP 440)
      python_requires='>=2.7, <4',
      packages=find_packages(),
      # Registered namespace vs. local directory
      package_dir={'PreFRBLE': "PreFRBLE"},
      include_package_data=True,        # Include non-Python files
      install_requires=requirements,    # Parse in list of requirements
      zip_safe=False,                   # Do not zip the installed package
      )
