#!/usr/bin/env python

#from distutils.core import setup
import pharaglow
from setuptools import setup

if __name__ == "__main__":
      setup(name='pharaglow',
            version=pharaglow.__version__,
            description='a toolset to analyze videos of foraging animals. ',
            author='M.Scholz',
            author_email='monika.k.scholz@gmail.com',
            packages=['pharaglow'],
            install_requires=[
                  "numpy",
                  "scipy",
                  "pyampd",
                  "pandas",
                  "pims",
                  "scikit-image",
                  "trackpy",
            ],
      )
