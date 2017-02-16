#!/usr/bin/env python
from setuptools import setup, Extension, find_packages
import sys

import numpy as np

long_description = ''

if 'upload' in sys.argv:
    with open('README.rst') as f:
        long_description = f.read()

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Natural Language :: English',
    'Programming Language :: C',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: Implementation :: CPython',
    'Topic :: Scientific/Engineering',
]

setup(
    name='warp_prism',
    version='0.1.0',
    description='Quickly move data from postgres to numpy or pandas.',
    author='Quantopian Inc.',
    author_email='opensource@gmail.com',
    packages=find_packages(),
    long_description=long_description,
    license='Apache 2.0',
    classifiers=classifiers,
    url='https://github.com/quantopian/warp_prism',
    ext_modules=[
        Extension(
            'warp_prism._warp_prism',
            ['warp_prism/_warp_prism.c'],
            include_dirs=[np.get_include()],
        ),
    ],
    install_requires=[
        'numpy',
        'pandas',
        'sqlalchemy',
        'psycopg2',
        'toolz',
    ],
    extras_require={
        'odo': ['odo'],
    },
)
