#!/usr/bin/env python
from setuptools import setup, Extension, find_packages
import sys

import numpy as np

long_description = ''

if 'upload' in sys.argv:
    with open('README.rst') as f:
        long_description = f.read()

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Natural Language :: English',
    'Programming Language :: C',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: Implementation :: CPython',
    'Topic :: Scientific/Engineering',
]

setup(
    name='warp_prism',
    version='0.1.1',
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
            extra_compile_args=['-std=c99', '-Wall', '-Wextra'],
        ),
    ],
    install_requires=[
        'datashape',
        'numpy',
        'pandas',
        'sqlalchemy',
        'psycopg2',
        'odo',
        'toolz',
    ],
    extras_require={
        'dev': [
            'flake8==3.3.0',
            'pycodestyle==2.3.1',
            'pyflakes==1.5.0',
            'pytest==3.0.6',
        ],
    },
)
