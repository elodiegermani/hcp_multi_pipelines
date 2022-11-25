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

    random_subject_list = []
    for i in range(n_iter):
        random_subject_list.append(np.random.choice(subject_list, n_sub, True)) # Creates a list of list containing n random subjects

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    ## output_dir : where the final results will be store
    output_dir = f"group_maps"

    ## working_dir : where the intermediate results will be stored
    working_dir = f"intermediate_results"

    l2_analysis = group_analysis.get_l2_analysis(exp_dir, output_dir, working_dir, result_dir,
        subject_list, contrast_list, gzip)
    l2_analysis.run('MultiProc', plugin_args={'n_procs': 16}) 
