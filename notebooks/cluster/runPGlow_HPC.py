import papermill as pm
import os
import sys
import json
from multiprocessing import Pool
import asyncio
import platform
import argparse


# worker function for analysing a single notebook
def work(pars):
    try:
        # remove unneccesary params
        nbook = pars.pop('templateNotebook')
        pars.pop('batchPath')
        print(pars)
        jobname = os.path.join(pars['outPath'], 'out_{}.ipynb'.format(pars['movie']))
        print(f'Analyzing {pars["movie"]}. Output can be watched live in {jobname}')
        pm.execute_notebook(
            nbook,
            os.path.join(pars['outPath'], f'out_{pars["movie"]}.ipynb'),
            parameters=pars
        )
    # skips to next movie if error raised to streamline analysis process
    except pm.exceptions.PapermillExecutionError:
        print(pars['movie'], 'ERROR')
        pass


def main(parfile, nworkers=1, mock = False, single = False):
    """run batch jupyter notebook analysis.

        mock (bool): if True, create jobs but don't run the analyses.
        nworkers (int): if >1, this will use multiprocessing to analyse multiple jobs concurrently.
    """
    if platform.system() =='Windows':
        sys.stdout.write("Windows operating system detected, Swithing to asyncio. \n")
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # get the batch parameters
    if os.path.isfile(parfile):
        with open(parfile) as f:
            # parameters handed to each notebook -- customize
            pars = json.load(f)
    else:
        sys.stdout.write('Parameter file is not a file or file not found.\n')
        sys.exit()

    jobs = []

    # create a dictionary of parameters
    if single:
        pars['inPath'] = pars['batchPath']
        pars['movie'] = movie = pars['inPath'].split('/')[-1]
        jobs.append(pars)
    else:
        for subfolder in [f.path for f in os.scandir(pars['batchPath']) if f.is_dir()]:
            movie = subfolder.split('/')[-1]
            if not movie.startswith('.'):
                pars['inPath'] = subfolder
                pars['movie'] = movie
                jobs.append(pars)

    # analysis via papermill
    if mock:
        sys.stdout.write(f'Created {len(jobs)} jobs for analysis. Mock run only, no analysis.\n')
        sys.exit()

    if nworkers > 1:
        # run multiple
        p = Pool(processes = nworkers)
        for i, _ in enumerate(p.imap_unordered(work, jobs)):
            sys.stdout.write('\rdone {0:%}'.format(i/len(jobs)))
        p.close()
        p.join()
    else:
        # run one-by-one
        for i,j in enumerate(jobs):
            work(j)
            sys.stdout.write('\rdone {0:%}'.format(i/len(jobs)))
    sys.stdout.write(f'Script finished. Completed {len(jobs)} jobs. \n')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("parfile", type=str,
                        help="JSON parameter file containing information about the jobs to be run.")
    parser.add_argument("-m", "--mock", type=bool,default = False,
                        help="run a mock analysis without actually starting the jobs.")
    parser.add_argument("-n", "--nworkers", type=int, default = 1,
                        help="Number of multiprocessing processes to use for parallelization. If n=1, run serially..")
    parser.add_argument("-s", "--single", type=bool, default = False,
                        help="Analyze a single directory.")
    args = parser.parse_args()
    main(args.parfile, args.nworkers, args.mock, args.single)
   