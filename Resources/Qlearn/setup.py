#!/usr/bin/env python3
from setuptools import setup, find_packages
import sys
import re

def parse_requirements():
    requires = []
    with open('requirements.txt', 'r') as fr :
        for line in fr :
            pkg = line.strip()
            requires.append(pkg)
    return requires


with open('qmlearn/__init__.py') as fd :
    lines = fd.read()
    __version__ = re.search('__version__ = "(.*)"', lines).group(1)
    __author__ = re.search('__author__ = "(.*)"', lines).group(1)
    __contact__ = re.search('__contact__ = "(.*)"', lines).group(1)
    __license__ = re.search('__license__ = "(.*)"', lines).group(1)

assert sys.version_info >= (3, 6)
description = "QMLearn"

with open('README.md') as fh :
    long_description = fh.read()

scripts=[]

extras_require = {
        'all' : [
            'matplotlib',
            ],
        }

setup(name='qmlearn',
      description=description,
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='http://qmlearn.rutgers.edu',
      version=__version__,
      # use_scm_version={'version_scheme': 'post-release'},
      # setup_requires=['setuptools_scm'],
      author=__author__,
      author_email=__contact__,
      license=__license__,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      packages=find_packages(),
      scripts=scripts,
      include_package_data=True,
      extras_require = extras_require,
      install_requires= parse_requirements())
