import papermill as pm
import os
import sys
import json
from multiprocessing import Pool
import asyncio
import platform
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ignoring the black dependency warning issued by papermill
class IgnoreBlackWarning(logging.Filter):
    def filter(self, record):
        return 'Black is not installed' not in record.msg
logging.getLogger("papermill.translators").addFilter(IgnoreBlackWarning())

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
            parameters=pars, kernel_name=pars["kernel_name"]
        )
    # skips to next movie if error raised to streamline analysis process
    except pm.exceptions.PapermillExecutionError as pmex:
        print(f"PapermillExecutionError ocurred processing movie: {pars['movie']}")
        logger.exception(pmex)
        pass


def main(parfile, nworkers=1, mock = False, single = False, kernel_name = 'pharaglow'):
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

    # setting the ipython kernel used to execute the jupyter notebooks
    # escaping awk curly braces (https://docs.python.org/2/library/string.html#format-string-syntax)
    stream = os.popen(f"jupyter kernelspec list | grep {kernel_name} | head -n 1 | awk '{{print $1}}'")
    kernel_test_output = stream.read().strip()
    if kernel_test_output != kernel_name:
        sys.stderr.write(f"The ipython kernel '{kernel_name}' is missing.\n")
        sys.exit()
    else:
        pars['kernel_name'] = kernel_name

    # creating an array of jobs (Python multiprocessing)
    jobs = []

    # create a dictionary of parameters
    if single:
        if pars['batchPath'].endswith('/'):
            pars['batchPath'] = pars['batchPath'][:-1]
        pars['inPath'] = pars['batchPath']
        pars['movie'] = os.path.basename(pars['batchPath'])
        jobs.append(pars)
    else:
        for subfolder in [f.path for f in os.scandir(pars['batchPath']) if f.is_dir()]:
            movie = subfolder.split('/')[-1]
            if not movie.startswith('.'):
                pars['inPath'] = subfolder
                pars['movie'] = movie
                jobs.append(pars.copy())

    # analysis via papermill
    if mock:
        for j in jobs:
            sys.stdout.write(f"{j['movie']}:{j['inPath']}\n")
        sys.stdout.write(f'Created {len(jobs)} jobs for analysis. Mock run only, no analysis.\n')
        sys.exit()

    if nworkers > 1:
        # run multiple
        p = Pool(processes = nworkers)
        for i, _ in enumerate(p.imap_unordered(work, jobs)):
            sys.stdout.write('\rdone {0:%}\n'.format(i/len(jobs)))
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
    parser.add_argument("-k", "--kernel", type=str, default='pharaglow',
                        help="The name of the ipython kernel where pharaglow is installed.")
    args = parser.parse_args()
    main(args.parfile, args.nworkers, args.mock, args.single, args.kernel)
