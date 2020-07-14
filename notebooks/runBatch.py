import papermill as pm
import os

parameterfile = "/home/nzjacic/Desktop/pharaglow_parameters_mks_batch.txt"
lawnPath = "/opt/data/Lawns/"
dataFolder = "/home/nzjacic/Desktop/Harddrive/10x_GRU101_nolawn/No_smell/Test/"
batchFolder = "/home/nzjacic/Desktop/Harddrive/10x_GRU101_nolawn/No_smell/Test/"

templateNotebook = '/home/nzjacic/Desktop/BatchRunTemplate-NZ.ipynb'
# create a dictionary of parameters
for subfolder in [f.path for f in os.scandir(dataFolder) if f.is_dir()]:
    movie = subfolder.split('/')[-1]
    if not movie.startswith('.'):
        pars = { 'parameterfile': parameterfile,
            'inPath': subfolder,
            'outPath': batchFolder,
            'lawnPath': lawnPath,
            'movie': movie,
            'nWorkers': 5,
        }
        print('Analyzing {}. Output can be watched live in'.format(movie), os.path.join(batchFolder, 'out_{}.ipynb'.format(movie)))
        try:
            pm.execute_notebook(
               templateNotebook,
               os.path.join(batchFolder, 'out_{movie}.ipynb'),
               parameters=pars
           )
        # skips to next movie if error raised to streamline analysis process
        except pm.exceptions.PapermillExecutionError:
            print(movie, 'ERROR')
            pass
