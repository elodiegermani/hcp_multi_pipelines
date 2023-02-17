#python3
# This script performs group maps computation between two datasets.
# Usage: python3 -g1 /srv/tempdd/egermani/hcp_pipelines/data/derived/dataset1 -g2 /srv/tempdd/egermani/hcp_pipelines/data/derived/dataset2 -S '["100206", ...]' -c '["rh"]' -r /srv/tempdd/egermani/hcp_pipelines/figures -i 1000

import numpy as np
import nibabel as nib 
from lib import between_group_analysis
from glob import glob
from os.path import join as opj
import os
import sys
import getopt
import json
import warnings
import csv

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=RuntimeWarning)

def compute_group_comparison(exp_dir_group1, exp_dir_group2, result_dir, subject_list, contrast_list, gzip=[True, True]):
    """
    Function to run Nipype workflow corresponding to group comparisons. 
    Parameters:
        - exp_dir_group1: str, path to directory where to find files from group1
        - exp_dir_group2: str, path to directory where to find files from group2
        - result_dir: str, path to directory where to store results and intermediate results
        - subject_list: list of list of str, lists of subjects to use for each iteration
        - contrast_list: list of str, list of contrast to which perform analysis 
        - gzip: list of Bool, perform gunzip or not on file for group1 and 2 (depend on wether files are already unzipped or not)
    """  

    # Important directories
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    ## working_dir : where the intermediate outputs will be store
    working_dir = f"intermediate_results"
    ## output_dir : where the final results will be store
    output_dir = f"final_results_group_comparison"

    # Create workflow and run it
    l2_analysis_generated = group_analysis.get_l2_analysis_group_comparison(exp_dir_group1, exp_dir_group2, output_dir, working_dir, result_dir, 
        subject_list, contrast_list, gzip=gzip)
    l2_analysis_generated.run('MultiProc', plugin_args={'n_procs': 16})
            
if __name__ == "__main__":
    exp_dir_group1 = None
    exp_dir_group2 = None
    subject_list = None
    contrast_list = None
    result_dir = None
    n_iter = None 

    try:
        OPTIONS, REMAINDER = getopt.getopt(sys.argv[1:], 'g1:g2:S:c:r:i', ['exp_dir_group1=', 'exp_dir_group2=',
            'subject_list=', 'contrast_list=', 'result_dir=', 'n_iter='])

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    # Replace variables depending on options
    for opt, arg in OPTIONS:
        if opt in ('-g1', '--exp_dir_group1'):
            exp_dir_group1= str(arg)
        elif opt in ('-g2', '--exp_dir_group2'):
            exp_dir_group2 = str(arg)
        elif opt in ('-S', '--subject_list'): 
            subject_list = json.loads(arg)
        elif opt in ('-c', '--contrast_list'): 
            contrast_list = json.loads(arg)
        elif opt in ('-r', '--result_dir'):
            result_dir = str(arg)
        elif opt in ('-i', '--n_iter'):
            n_iter = int(arg)


    print('OPTIONS   :', OPTIONS)

    gzip = [True, True]
    # If SPM files, already unziped so no need to re-unzip them during pipeline
    if 'SPM' in exp_dir_group1:
        gzip[0] = False
    if 'SPM' in exp_dir_group2:
        gzip[1] = False
    
    # If file containing list of groups doesn't exist, create it with random groups
    if not os.path.exists(opj(('/').join(result_dir.split('/')[:-1]), 'groups.csv')):
        random_subject_list = []
        for i in range(n_iter):
            random_subject_list.append(np.random.choice(subject_list, 100, False))

        with open(opj(('/').join(result_dir.split('/')[:-1]), 'groups.csv'), 'w') as file:
            for i, sub_list in enumerate(random_subject_list):
                for j, sub in enumerate(sub_list):
                    file.write(str(sub))
                    if j != 99:
                        file.write(',')
                file.write('\n')
        file.close()

    else: # Read it if it exists
        with open(opj(('/').join(result_dir.split('/')[:-1]), 'groups.csv'), 'r') as file:
            reader = csv.reader(file)
            random_subject_list = list(reader)
        file.close()
        print(random_subject_list)

    compute_group_comparison(exp_dir_group1, exp_dir_group2, result_dir, random_subject_list, contrast_list, gzip=gzip)
