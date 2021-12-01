#!/usr/bin/env python

from distutils.core import setup
import pharaglow

setup(name='pharaglow',
      version=pharaglow.__version__,
      description='a toolset to analyze videos of foraging animals. ',
      author='M.Scholz',
      author_email='monika.k.scholz@gmail.com',
      packages=['pharaglow'],
     )
