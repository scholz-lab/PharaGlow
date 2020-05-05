import papermill as pm
import os

parameterfile = "/home/nzjacic/Desktop/pharaglow_parameters_mks.txt"
lawnPath = "/media/scholz_la/hd2/Nicolina/Lawns"
dataFolder = "/media/scholz_la/hd2/Nicolina/Raw_videos/10x_INF100_test"

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
       '/home/nzjacic/Desktop/Code/PharaGlow/notebooks/BatchRunTemplate.ipynb',
       os.path.join(dataFolder, 'out_{movie}.ipynb'),
       parameters=pars
   )
