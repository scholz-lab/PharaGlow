import papermill as pm
import os

parameterfile = "D:/pharaglow_parameters_mks.txt"
lawnPath = "D:/Lawns"
dataFolder = "D:/10x_INF103"

# create a dictionary of parameters
for subfolder in [f.path for f in os.scandir(dataFolder) if f.is_dir()]:
    movie = subfolder.split('/')[-1]
    if not movie.startswith('.'):
        pars = { 'parameterfile': parameterfile,
            'inPath': subfolder,
            'outPath': dataFolder,
            'lawnPath': lawnPath,
            'movie': movie,
            'nWorkers': 5,
        }
        print('Analyzing {}. Output can be watched live in'.format(movie), os.path.join(dataFolder, 'out_{}.ipynb'.format(movie)))
        pm.execute_notebook(
           '/home/nzjacic/Desktop/Code/PharaGlow/notebooks/BatchRunTemplate.ipynb',
           os.path.join(dataFolder, 'out_{movie}.ipynb'),
           parameters=pars
       )
