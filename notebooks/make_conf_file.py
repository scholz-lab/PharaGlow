# Creates pharaglow batch config file using path provided as script arguments
# NOTE: all the paths are referred to the root_dir provided as an argument.
#
# USAGE EXAMPLE: 
# python make_conf_file.py  -d $HOME/pharaglow  \
#                           -i data -o out_data \
#                           -t notebooks -n PharaGlowHPC.ipynb \
#                           -c pglow_batch_config_github.json

import os
import sys
import json
import argparse

def make_config(root_dir,
                config_name = 'pglow_batch_config_local.json', 
                param_file = 'AnalysisParameters_1x.json', 
                input_dir = 'data', 
                output_dir = 'out_data',
                templates_dir = 'notebooks',
                template_name = 'PharaGlowHPC.ipynb', 
                nworkers = 10, 
                lawn_dir = None, 
                depth = 'uint8', 
                save_minimal = True):

    pglow_config = {
        'parameterfile': os.path.join(root_dir, param_file),
        'batchPath': os.path.join(root_dir, input_dir),
        'outPath': os.path.join(root_dir, output_dir),
        'templateNotebook': os.path.join(root_dir, templates_dir, template_name),
        'nWorkers': nworkers,
        'lawnPath': os.path.join(root_dir, lawn_dir) if lawn_dir else 'None', #null#
        'depth': depth,
        'save_minimal': save_minimal
    }

    # serializing pharaglow config to json
    pglow_config_json = json.dumps(pglow_config, indent=4)

    # writing pharaglow config to templates directory
    config_file = os.path.join(templates_dir, config_name) 
    with open (config_file, 'w') as outfile:
        outfile.write(pglow_config_json)
        sys.stdout.write(f'[INFO] pharaglow batch config file created at: {config_file}\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-n", "--tplname", type=str, default= 'PharaGlowHPC.ipynb',
                        help="pharaglow template notebook.")
    parser.add_argument("-d", "--rootdir", type=str,
                        help="pharaglow root directory.") #NOTE this argument is mandatory
    parser.add_argument("-c", "--config", type=str, default = 'pglow_batch_config_local.json',
                        help="JSON config file name.")
    parser.add_argument("-p", "--parfile", type=str, default = 'AnalysisParameters_1x.json',
                        help="JSON parameter file containing information about the jobs to be run.")
    parser.add_argument("-i", "--inputdir", type=str,
                        help="pharaglow data input directory.")
    parser.add_argument("-o", "--outputdir", type=str,
                        help="pharaglow data output directory.")
    parser.add_argument("-t", "--templatedir", type=str,
                        help="templates directory.")
    parser.add_argument("-w", "--workers", type=int, default = 10,
                        help="Number of workers used for parallelization.")
    parser.add_argument("-l", "--lawndir", type=str,
                        help="pharaglow batch output lawn directory.")
    parser.add_argument("-e", "--depth", type=str, default='uint8',
                        help="pharaglow depth") #NOTE: review name
    parser.add_argument("-s", "--savemin", type=bool, default=True,
                        help="Sets pharaglow batch save_minimal flag.") 
    

    args = parser.parse_args()

    make_config(
        args.rootdir,
        args.config,
        args.parfile,
        args.inputdir,
        args.outputdir,
        args.templatedir,
        args.tplname,
        args.workers,
        args.lawndir,
        args.depth,
        args.savemin 
    )
