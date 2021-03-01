#!/usr/bin/env python

"""io.py: loading for pharaglow feature files."""
import numpy as np
import pandas as pd
import warnings

def load(fname, image_depth =8, maxcols = 10000, prefix = "im", **kwargs):
    """load a pharglow features, trajectories or results file.
        We expect columns containing the string 'im' to be image pixels which will convert to uint8.
    """
    converter = {}
    for i in range(maxcols):
        converter[f'im{i}']= 'uint8'
    
    return pd.read_json(fname, dtype = converter, **kwargs)
