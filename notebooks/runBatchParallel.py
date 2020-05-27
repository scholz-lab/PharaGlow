import papermill as pm
import os
import time
from multiprocessing import Pool, Process
import asyncio
import platform
import sys

# this needn\t be changed unless using a different batch notebook
notebookPath = 'D:/Code/PharaGlow/notebooks/BatchRunTemplate.ipynb'
if platform.system() =='Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# worker function for parallelization
def work(pars):
    print(pars['outPath'])
    print('Analyzing {}. Output can be watched live in'.format(pars['movie']), os.path.join(pars['outPath'], 'out_{}.ipynb'.format(pars['movie'])))
    pm.execute_notebook(
           notebookPath,
           os.path.join(pars['outPath'], 'out_{}.ipynb'.format(pars['movie'])),
           parameters=pars)


def main():
    parameterfile = "D:/pharaglow_parameters_mks.txt"
    lawnPath = "D:/Lawns"
    dataFolder = "D:/10x_INF103"

    jobs = []
    # create a dictionary of parameters and make jobs for the parallelizer
    for subfolder in [f.path for f in os.scandir(dataFolder) if f.is_dir()]:
        print(subfolder)
        movie = os.path.basename(subfolder)
        if not movie.startswith('.'):
            pars = { 'parameterfile': parameterfile,
                'inPath': subfolder,
                'outPath': dataFolder,
                'lawnPath': lawnPath,
                'movie': movie,
                'nWorkers': 1,
                 }
        jobs.append(pars)

    p = Pool(processes = 12)
    start = time.time()
    for i, _ in enumerate(p.imap_unordered(work, jobs)):
       sys.stderr.write('\rdone {0:%}'.format(i/len(jobs)))
    p.close()
    p.join()



if __name__ == "__main__":
    # run multiple jobs in parallell
    main()
    
        
       
