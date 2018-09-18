#!/usr/bin/env python
# -*- coding: utf-8 -*

import os

from setuptools import find_packages, setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()
    install_requires = [i for i in install_requires if '://' not in i]

VERSION = '0.0.1.DEV0'

setup(
    name='scikit-keras',
    version=VERSION,
    packages=find_packages(exclude=('tests',)),
    include_package_data=True,
    zip_safe=False,
    description='Compatibility wrapper for keras models',
    author='Igor Gotlibovych',
    author_email='igor.',
    license='MIT',
    install_requires=install_requires,
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
