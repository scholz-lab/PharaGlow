import papermill as pm
import os

parameterfile = "/home/mscholz/Desktop/Code/TestDataPGlow/exampleParameterfile.json"
lawnPath = None
dataFolder = "/home/mscholz/Desktop/Code/TestDataPGlow"

# create a dictionary of parameters
for subfolder in [f.path for f in os.scandir(dataFolder) if f.is_dir()]:
    movie = subfolder.split('/')[-1]
    
    pars = { 'parameterfile': parameterfile,
            'inPath': subfolder,
            'outPath': dataFolder,
            'lawnPath': lawnPath,
            'movie': movie
    }

    pm.execute_notebook(
       '/home/mscholz/Desktop/Code/PharaGlow/notebooks/BatchRunTemplate.ipynb',
       '/home/mscholz/Desktop/Code/PharaGlow/notebooks/out_{movie}.ipynb',
       parameters=pars
   )