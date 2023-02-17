#python3
# This script contains functions to perform two-sample-t-tests between two groups 
# with groups defined as a list of subjects with the 0-n subjects corresponding to group1 and n+1-end to group2.

from nipype.interfaces import spm
matlab_cmd = '/opt/spm12-r7771/run_spm12.sh /opt/matlabmcr-2010a/v713/ script'
spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

from nipype.interfaces.spm import (Coregister, Smooth, OneSampleTTestDesign, EstimateModel, EstimateContrast, 
                                   Level1Design, TwoSampleTTestDesign, Realign, 
                                   Normalize12, NewSegment)
from nipype.interfaces.fsl import ExtractROI, Info
#from nipype.interfaces.spm import ThresholdStatistics
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip
from nipype import Workflow, Node, MapNode, JoinNode
from nipype.interfaces.base import Bunch

from os.path import join as opj
import os
import json

def get_groups_maps(group1_files, group2_files, subject_list, n):
    ''' 
    Functions to get corresponding files depending on subject list to use. 
    Parameters:
        - group1_files: list of str representing existing files, all files for group 1
        - group2_files: list of str representing existing files, all files for group 2
        - subject_list: list of list of str, lists of subjects to use for each iteration
        - n: int, iteration to make

    Outputs:
        - group1_sublist: list of str representing existing files, files corresponding to selected subjects for group1
        - group2_sublist: list of str representing existing files, files corresponding to selected subjects for group2
    '''
    group1_sublist = []
    group2_sublist = []

    group1_subjects = subject_list[n][:int(len(subject_list[n])/2)] # Select the n-th list of subject and split it to obtain the two groups
    group2_subjects = subject_list[n][int(len(subject_list[n])/2):]

    for file in group1_files:
        sub_id = file.split('/')[-1].split('_')[-3]
        if sub_id in group1_subjects:
            group1_sublist.append(file)  # Get the files corresponding to subjects from the first part of the n-th list of subjects 

    for file in group2_files:
        sub_id = file.split('/')[-1].split('_')[-3]
        if sub_id in group2_subjects:
            group2_sublist.append(file) 

    print(group1_sublist)
    
    return group1_sublist, group2_sublist

def get_threshold_images(threshold, image):
    from nilearn.image import threshold_img
    import nibabel as nib 
    import os

    nii_img = nib.load(image)
    nii_img_thresh = threshold_img(nii_img, threshold)

    f_path = os.path.dirname(image) + os.path.basename(image).split('.')[0] + 'threshold_bonferroni.nii'
    nib.save(nii_img_thresh, f_path)

    return f_path

def get_l2_analysis_group_comparison(exp_dir_group1, exp_dir_group2, output_dir, working_dir, result_dir, subject_list, contrast_list, gzip=[True, True]): 
    """
    Function to create Nipype workflow corresponding to group comparisons. 
    Parameters:
        - exp_dir_group1: str, path to directory where to find files from group1
        - exp_dir_group2: str, path to directory where to find files from group2
        - output_dir: str, name of the directory where to store results
        - working_dir: str, name of the directory where to store intermediate results
        - result_dir: str, path to directory where to store results and intermediate results
        - subject_list: list of list of str, lists of subjects to use for each iteration
        - contrast_list: list of str, list of contrast to which perform analysis 
        - gzip: list of Bool, perform gunzip or not on file for group1 and 2 (depend on wether files are already unzipped or not)

    Outputs:
        - l2_analysis: Nipype workflow
    """         
    # Infosource - a function free node to iterate over the list of subject names
    infosource_groupanalysis = Node(IdentityInterface(fields=['contrast', 'task']),
                      name="infosource_groupanalysis")

    infosource_groupanalysis.iterables = [('contrast', contrast_list)]

    # SelectFiles templates and Node
    group1_files = opj(exp_dir_group1, 'sub_*_contrast_{contrast}.nii*')
    group2_files = opj(exp_dir_group2, 'sub_*_contrast_{contrast}.nii*')

    templates = {'group1' : group1_files, 'group2':group2_files}
    
    selectfiles_groupanalysis = Node(SelectFiles(templates, base_directory=result_dir, force_list= True),
                       name="selectfiles_groupanalysis")
    
    # Datasink node : to save important files 
    datasink_groupanalysis = Node(DataSink(base_directory = result_dir, container = output_dir), 
                                  name = 'datasink_groupanalysis')

    
    # Node to select subset of files corresponding to selected subjects
    sub_contrasts = Node(Function(input_names = ['group1_files', 'group2_files', 'subject_list', 'n'],
                                 output_names = ['group1_sublist', 'group2_sublist'],
                                 function = get_groups_maps),
                        name = 'sub_contrasts')

    sub_contrasts.iterables = ('n', range(0, len(subject_list))) # Iterate on all lists of subjects 
    sub_contrasts.inputs.subject_list = subject_list
    
    # Node for the design matrix
    two_sample_t_test_design = Node(TwoSampleTTestDesign(unequal_variance=True), name = 'two_sample_t_test_design')

    # Estimate model 
    estimate_model = Node(EstimateModel(estimation_method={'Classical':1}), name = "estimate_model")

    # Estimate contrasts
    estimate_contrast = Node(EstimateContrast(group_contrast=True),
                             name = "estimate_contrast")

    # Define contrasts to analyse 
    contrasts = [('1 > 2', 'T', ['Group_{1}', 'Group_{2}'], [1, -1]), 
                    ('2 > 1', 'T', ['Group_{1}', 'Group_{2}'], [-1, 1])]

    estimate_contrast.inputs.contrasts = contrasts

    #threshold = MapNode(ThresholdStatistics(height_threshold=0.001), name='threshold', iterfield=['contrast_index', 'stat_image'])

    #threshold.inputs.contrast_index = [1,2]

    #threshold.synchronize = True

    #threshold_img = Node(Function(input_names=['threshold', 'image'], 
    #    output_names = ['nii_img_thresh'], function = get_threshold_images), name = 'threshold_img')

    # Create Workflow and make connections
    l2_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = f"l2_analysis")

    l2_analysis.connect([(infosource_groupanalysis, selectfiles_groupanalysis, [('contrast', 'contrast')]),
        (selectfiles_groupanalysis, sub_contrasts, [('group1', 'group1_files'), ('group2', 'group2_files')]),
        (two_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')]),
        (estimate_model, estimate_contrast, [('spm_mat_file', 'spm_mat_file'),
            ('residual_image', 'residual_image'),
            ('beta_images', 'beta_images')]),
        (estimate_model, datasink_groupanalysis, [('mask_image', f"l2_analysis.@mask")]),
        (estimate_contrast, datasink_groupanalysis, [('spm_mat_file', f"l2_analysis.@spm_mat"),
            ('spmT_images', f"l2_analysis.@T"),
            ('con_images', f"l2_analysis.@con")])])

        #(estimate_contrast, threshold, [('spm_mat_file', 'spm_mat_file'), 
        #    ('spmT_images', 'stat_image')]), 
        #(threshold, threshold_img, [('voxelwise_P_Bonf', 'threshold')]), 
        #(estimate_contrast, threshold_img, [('spmT_images', 'image')]), 
        #(threshold_img, datasink_groupanalysis, [('nii_img_thresh', f'l2_analysis.@threshold_img')])])
    
    if gzip[0]: # If files from group1 are .gz, gunzip them before input to node 
        # Gunzip Node if .gz files
        gunzip_group1 = MapNode(Gunzip(), name = 'gunzip_group1', iterfield=['in_file'])
        l2_analysis.connect([(sub_contrasts, gunzip_group1, [('group1_sublist', 'in_file')]), 
            (gunzip_group1, two_sample_t_test_design, [("out_file", 'group1_files')])])
    else: # Else, input to node directly
        l2_analysis.connect([(sub_contrasts, two_sample_t_test_design, [("group1_sublist", 'group1_files')])])

    if gzip[1]: # Idem for group2
        gunzip_group2 = MapNode(Gunzip(), name = 'gunzip_group2', iterfield=['in_file'])
        l2_analysis.connect([(sub_contrasts, gunzip_group2, [('group2_sublist', 'in_file')]), 
            (gunzip_group2, two_sample_t_test_design, [("out_file", 'group2_files')])])
    else:
        l2_analysis.connect([(sub_contrasts, two_sample_t_test_design, [('group2_sublist', 'group2_files')])])

    return l2_analysis