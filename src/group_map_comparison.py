#python3
#This script runs a group analysis on a dataset, i times for groups of n subjects
import numpy as np
import nibabel as nib 
from lib import group_analysis
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

if __name__ == "__main__":
    exp_dir = None
    subject_list = None
    contrast_list = None
    result_dir = None
    n_iter = None 

    try:
        OPTIONS, REMAINDER = getopt.getopt(sys.argv[1:], 'e:S:c:r:i:n', ['exp_dir=', 
            'subject_list=', 'contrast_list=', 'result_dir=', 'n_iter=', 'n_sub='])

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    # Replace variables depending on options
    for opt, arg in OPTIONS:
        if opt in ('-e', '--exp_dir'): # Directory where original data are stored (subject level data)
            exp_dir= str(arg)
        elif opt in ('-S', '--subject_list'): # Global subject list in which randomly sampling the n subjects
            subject_list = json.loads(arg)
        elif opt in ('-c', '--contrast_list'): # List of contrasts on which to do the analysis
            contrast_list = json.loads(arg)
        elif opt in ('-r', '--result_dir'): # Directory where derived data will be stored
            result_dir = str(arg)
        elif opt in ('-i', '--n_iter'): # Number of times to perform the group analysis (number of different subject lists)
            n_iter = int(arg)
        elif opt in ('-n', '--n_sub'): # Number of subjects per group
            n_sub = int(arg)

    print('OPTIONS   :', OPTIONS)

    gzip = True
    if 'SPM' in exp_dir:
        gzip = False # SPM files are already unzipped

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # If file containing list of groups doesn't exist, create it with random groups
    if not os.path.exists(opj(('/').join(result_dir.split('/')[:-1]), 'groups.csv')):
        random_subject_list = []
        for i in range(n_iter):
            random_subject_list.append(np.random.choice(subject_list, n_sub, False))

        with open(opj(('/').join(result_dir.split('/')[:-1]), 'groups.csv'), 'w') as file:
            for i, sub_list in enumerate(random_subject_list):
                for j, sub in enumerate(sub_list):
                    file.write(str(sub))
                    if j != n_sub-1:
                        file.write(',')
                file.write('\n')
        file.close()

    else: # Read it if it exists
        with open(opj(('/').join(result_dir.split('/')[:-1]), 'groups.csv'), 'r') as file:
            reader = csv.reader(file)
            random_subject_list = list(reader)
        file.close()

    ## output_dir : where the final results will be store
    output_dir = f"group_maps"

    ## working_dir : where the intermediate results will be stored
    working_dir = f"intermediate_results"

    print(random_subject_list)

    l2_analysis = group_analysis.get_l2_analysis(exp_dir, output_dir, working_dir, result_dir,
        random_subject_list, contrast_list, gzip)
    l2_analysis.run('MultiProc', plugin_args={'n_procs': 16}) 
