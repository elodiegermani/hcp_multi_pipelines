#python3
import pandas as pd
import numpy as np
from glob import glob 
import os
from os.path import join as opj
import shutil
from lib import preprocessing
import sys

def create_original_dataset_subject(dataset_template, dataset_config, contrasts=['cue', 'lf', 'lh', 'rf', 'rh', 't'], rm_existing=False):
    '''
    Performs original dataset creation (make directory, copy and rename files, creates txt file containing IDs)
    Works for SPM and FSL datasets obtained with hcp_pipelines scripts. 
    
    
    Parameters:
        - dataset_template: str, template for glob function (e.g. path to be able to find all data of the dataset)
        - dataset_config: dict, dictionnary with 3 keys (SOFT, FWHM and MC_PARAM) and values corresponding 
                                to the attributes of the dataset
        - contrasts: list, sorted list of contrasts used in the dataset
        - rm_existing: bool, whether to remove existing directory and overwrite existing data or not.
    '''
    # Dataset name for directory
    dataset_name = 'DATASET_'+'_'.join([str(i)+'_'+str(y) for i,y in dataset_config.items()])
    
    # Create directories or remove them if already exists
    if not os.path.exists(f'../data/derived/subject_level/{dataset_name}'):
        print('Creating dataset...')
        os.mkdir(f'../data/derived/subject_level/{dataset_name}')
        os.mkdir(f'../data/derived/subject_level/{dataset_name}/original')
    else:
        if rm_existing:
            print('Removing existing directory...')
            shutil.rmtree(f'../data/derived/subject_level/{dataset_name}')
            os.mkdir(f'../data/derived/subject_level/{dataset_name}')
            os.mkdir(f'../data/derived/subject_level/{dataset_name}/original')
        else: 
            print('Dataset already exist.')
            sys.exit(2)
    
    # Find all files of the dataset, rename them and copy them in the correct directory
    id_list=[]
    for file in sorted(glob(dataset_template)):
        if dataset_config['SOFT']=='SPM':
            new_id = 'sub_' + file.split('/')[-2].split('_')[-3] + '_contrast_' + contrasts[int(file[-5])-1]
            print('Copying file:', new_id)
            new_file = f'../data/derived/subject_level/{dataset_name}/original/{new_id}.nii'
        elif dataset_config['SOFT']=='FSL':
            new_id = 'sub_' + file.split('/')[-2].split('_')[-3] + '_contrast_' + contrasts[int(file.split('/')[-2].split('_')[2])-1]
            print('Copying file:', new_id)
            new_file = f'../data/derived/subject_level/{dataset_name}/original/{new_id}.nii.gz'
        id_list.append(new_id)
        shutil.copyfile(file, new_file)
    
    # Write ids in a txt file that will be used for creating a Dataset object
    with open(opj(f'../data/derived/subject_level/{dataset_name}/{dataset_name}_IDS.txt'), "a") as file:
        for ids in id_list:
            file.write(str(ids))
            file.write('\n')
    file.close()
    
def create_original_dataset_group(dataset_template, dataset_config, n_sub, contrasts=['cue', 'lf', 'lh', 'rf', 'rh', 't'], 
                            rm_existing=False):
    '''
    Performs original dataset creation (make directory, copy and rename files, creates txt file containing IDs)
    Works for SPM and FSL datasets obtained with hcp_pipelines scripts. 
    
    
    Parameters:
        - dataset_template: str, template for glob function (e.g. path to be able to find all data of the dataset)
        - dataset_config: dict, dictionnary with 3 keys (SOFT, FWHM and MC_PARAM) and values corresponding 
                                to the attributes of the dataset
        - n_sub: int, number of subjects in groups 
        - contrasts: list, sorted list of contrasts used in the dataset
        - rm_existing: bool, whether to remove existing directory and overwrite existing data or not.
    '''
    # Dataset name for directory
    dataset_name = 'DATASET_'+'_'.join([str(i)+'_'+str(y) for i,y in dataset_config.items()])
    
    if not os.path.exists(f'../data/derived/group_level/group_{n_sub}'):
        print('Creating dataset...')
        os.mkdir(f'../data/derived/group_level/group_{n_sub}')
    
    # Create directories or remove them if already exists
    if not os.path.exists(f'../data/derived/group_level/group_{n_sub}/{dataset_name}/original'):
        print('Creating dataset...')
        os.mkdir(f'../data/derived/group_level/group_{n_sub}/{dataset_name}')
        os.mkdir(f'../data/derived/group_level/group_{n_sub}/{dataset_name}/original')
    else:
        if rm_existing:
            print('Removing existing directory...')
            shutil.rmtree(f'../data/derived/group_level/{dataset_name}')
            os.mkdir(f'../data/derived/group_level/group_{n_sub}/{dataset_name}')
            os.mkdir(f'../data/derived/group_level/group_{n_sub}/{dataset_name}/original')
            
    id_list= []
    
    # Find all files of the dataset, rename them and copy them in the correct directory
    for n, file in enumerate(sorted(glob(dataset_template))):
        n_iter = str(n)
        print(n_iter)
        new_id = 'n_'+ n_iter +'_contrast_' + file.split('/')[-3].split('_')[-1] 
        print('Copying file:', new_id)
        new_file = f'../data/derived/group_level/group_{n_sub}/{dataset_name}/original/{new_id}.nii'
        id_list.append(new_id)
        shutil.copyfile(file, new_file)
    
    # Write ids in a txt file that will be used for creating a Dataset object
    with open(opj(f'../data/derived/group_level/group_{n_sub}/{dataset_name}/{dataset_name}_IDS.txt'), "w") as file:
        for ids in id_list:
            file.write(str(ids))
            file.write('\n')
    file.close()