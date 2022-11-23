#python3
# This script performs error rate computation between two datasets.
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

def compute_error_rate(result_dir, n_iter, contrast_list):
    '''
    Compute error rate between the two groups by taking the percentage of images having at least one active voxel among the n_iter images.

    Parameters:
        - result_dir: str, path to the directory where to find the results of the two sample t-tests between groups
        - n_iter: int, number of iterations performed (default=1000)
        - contrast_list: list of str, contrast for which to perform the calcul

    Output: 
        - ER_1: float, error rate for group1 > group2 analysis
        - ER_2: float, error rate for group2 > group1 analysis
    '''
    print('Computing error rate...')

    frac_1=[]
    frac_2=[]

    for contrast in contrast_list: 
        for i in range(n_iter):
            # Files for contrast and iteration
            stat_map_1 = opj(result_dir, f'final_results_group_comparison_999', 'l2_analysis', f'_contrast_{contrast}', 
                          f'_n_{i}', '_threshold0', 'spmT_0001_thr.nii')
            stat_map_2 = opj(result_dir, f'final_results_group_comparison_999', 'l2_analysis', f'_contrast_{contrast}',
                          f'_n_{i}', '_threshold1', 'spmT_0002_thr.nii')
            mask = opj(opj(result_dir, f'final_results_group_comparison_999', 'l2_analysis', f'_contrast_{contrast}',
                          f'_n_{i}', 'mask.nii'))
            # Remove NaNs
            stat_map_1_data = np.nan_to_num(nib.load(stat_map_1).get_fdata()) 
            stat_map_2_data = np.nan_to_num(nib.load(stat_map_2).get_fdata()) 

            mask_data = np.nan_to_num(nib.load(mask).get_fdata()) 

            # Apply mask
            stat_map_1_data = stat_map_1_data * mask_data
            stat_map_2_data = stat_map_2_data * mask_data

            # Vectorize
            stat_map_1_data = np.reshape(stat_map_1_data, -1)
            stat_map_2_data = np.reshape(stat_map_2_data, -1)

            # Search for activated voxels
            if np.any(stat_map_1_data != 0):
                print(f'Image {i} contains errors.')
                frac_1.append(1)
            if  np.any(stat_map_2_data != 0):
                print(f'Image {i} contains errors.')
                frac_2.append(1)
            
        #print(len(frac_1), len(frac_2))

    ER_1 = len(frac_1)/n_iter
    ER_2 = len(frac_2)/n_iter

    print(ER_1)
    print(ER_2)

    return ER_1, ER_2
            
if __name__ == "__main__":
    exp_dir_group1 = None
    exp_dir_group2 = None
    subject_list = None
    contrast_list = None
    result_dir = None
    n_iter = None 

    # Usage: python3 -g1 /srv/tempdd/egermani/hcp_pipelines/data/derived/dataset1 -g2 /srv/tempdd/egermani/hcp_pipelines/data/derived/dataset2 -S '["100206", ...]' -c '["rh"]' -r /srv/tempdd/egermani/hcp_pipelines/figures -i 1000
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
    if not os.exists(opj(('/').join(result_dir.split('/')[:-1]), 'groups.csv')):
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

    compute_group_comparison(exp_dir_group1, exp_dir_group2, result_dir, random_subject_list, contrast_list, gzip=gzip)
    
    compute_error_rate(result_dir, n_iter, contrast_list)
