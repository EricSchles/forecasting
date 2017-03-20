#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

description = 'Code for Doing Time Series Prediction'
with open('README.rst') as readme:
    long_description = readme.read()

setup(
    name = 'forecaster',
    version = '0.0.1',
    url = 'https://github.com/EricSchles/forecaster',
    download_url = "https://github.com/EricSchles/forecaster/tarball/0.0.1",
    license = 'GPLv3',
    description = description,
    long_description = long_description,
    author = 'Eric Schles',
    author_email = 'ericschles@gmail.com',
    install_requires = [
        'statsmodels',
        'scipy',
        'numpy',
        'pandas',
        'sklearn'
    ],
    packages = ['forecaster'],
    package_dir={'forecaster': 'forecaster'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Intended Audience :: Statisticians',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Office/Business :: Financial',
        'Topic :: Utilities',
    ],
)
