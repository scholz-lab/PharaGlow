#!/usr/bin/env python

"""io.py: loading for pharaglow feature files."""
import logging
import pandas as pd


def load(fname, image_depth =8, maxcols = 10000, prefix = "im", **kwargs):
    """load a pharglow features, trajectories or results file.
        We expect columns containing the string 'im' to be image pixels which will convert to uint8.

    Args:
        fname (str): filename of the .json file to load.
        image_depth (int, optional): bit depth of the images. Defaults to 8.
        maxcols (int, optional): maximal number of expected columns. Defaults to 10000.
        prefix (str, optional): prefix to add to the column. Defaults to "im".

    Returns:
        pandas.DataFrame: a pharaglow file as dataframe
    """

    converter = {}
    for i in range(maxcols):
        converter[f'{prefix}{i}']= f'uint{image_depth}'

    return pd.read_json(fname, dtype = converter, **kwargs)


def log_setup(name, level, fname):
    """This function will setup a logger with the name and level you pass as input.
    Levels are 10 (debug), 20 (info), 30 (warning), 40 (error), 50 (critical).

    Args:
        name (str): name of the logger object
        level (int): logging level {10,20,30,40,50}
        fname (str): filename for writing the log messages

    Returns:
        logging.Logger: a logger
    """


    # start a logger
    logger = logging.getLogger(name)
    # set a formatter to manage the output format of our handler
    formatter = logging.Formatter('%(asctime)s | %(name)s |  %(levelname)s: %(message)s')

    # set the level passed as input, has to be logging.LEVEL not a string
    # until we do so mylog doesn't have a level and inherits the root logger level:WARNING
    logger.setLevel(level)

    # add a handler to send INFO level messages to console
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # logger.addHandler(console_handler)

    # add a handler to send DEBUG level messages to file
    # all you need is a file name I added the 'w' so each time a new file will be created
    # without it the messagges will be appended to the same file
    file_handler = logging.FileHandler(fname)
    # file_handler = logging.FileHandler(fname,'w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # return the logger object
    return logger
