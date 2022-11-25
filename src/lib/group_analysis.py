from nipype.interfaces import spm
matlab_cmd = '/opt/spm12-r7771/run_spm12.sh /opt/matlabmcr-2010a/v713/ script'
spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

from nipype.interfaces.spm import (Coregister, Smooth, OneSampleTTestDesign, EstimateModel, EstimateContrast, 
                                   Level1Design, TwoSampleTTestDesign, Realign, 
                                   Normalize12, NewSegment)
from nipype.interfaces.fsl import ExtractROI, Info
from nipype.interfaces.spm import Threshold 
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip
from nipype import Workflow, Node, MapNode, JoinNode
from nipype.interfaces.base import Bunch

from os.path import join as opj
import os
import json


def get_contrasts_maps(contrast_map, subject_list, i):
    ''' 
    '''
    contrast_map_sublist = []

    sub_subject_list = subject_list[i]

    for file in contrast_map:
        sub_id = file.split('/')[-1].split('_')[-3]
        if sub_id in sub_subject_list:
            contrast_map_sublist.append(file)  

    return contrast_map_sublist

def get_l2_analysis(exp_dir, output_dir, working_dir, result_dir, subject_list, contrast_list, gzip=True): 
    """
    """         
    # Infosource - a function free node to iterate over the list of subject names
    infosource_groupanalysis = Node(IdentityInterface(fields=['contrast']),
                      name="infosource_groupanalysis")

    infosource_groupanalysis.iterables = [('contrast', contrast_list)]

    # SelectFiles
    contrast_map_files = opj(exp_dir, 'sub_*_contrast_{contrast}.nii*')

    templates = {'contrast_map' : contrast_map_files}
    
    selectfiles_groupanalysis = Node(SelectFiles(templates, base_directory=result_dir, force_list= True),
                       name="selectfiles_groupanalysis")
    
    # Datasink node : to save important files 
    datasink_groupanalysis = Node(DataSink(base_directory = result_dir, container = output_dir), 
                                  name = 'datasink_groupanalysis')

    gunzip = MapNode(Gunzip(), name = 'gunzip', iterfield=['in_file'])
    
    # Node to select subset of contrasts
    sub_contrasts = Node(Function(input_names = ['contrast_map', 'subject_list', 'i'],
                                 output_names = ['contrast_map_sublist'],
                                 function = get_contrasts_maps),
                        name = 'sub_contrasts')

    sub_contrasts.iterables('i', range(0, len(subject_list)))
    n = len(subject_list[0])

    sub_contrasts.inputs.subject_list = subject_list

    ## Estimate model 
    estimate_model = Node(EstimateModel(estimation_method={'Classical':1}), name = "estimate_model")

    ## Estimate contrasts
    estimate_contrast = Node(EstimateContrast(group_contrast=True),
                             name = "estimate_contrast")

    l2_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = f"l2_analysis")

    l2_analysis.connect([(infosource_groupanalysis, selectfiles_groupanalysis, [('contrast', 'contrast')]),
        (selectfiles_groupanalysis, sub_contrasts, [('contrast_map', 'contrast_map')]),
        (estimate_model, estimate_contrast, [('spm_mat_file', 'spm_mat_file'),
            ('residual_image', 'residual_image'),
            ('beta_images', 'beta_images')]),
        (estimate_model, datasink_groupanalysis, [('mask_image', f"l2_analysis_n_{n}.@mask")]),
        (estimate_contrast, datasink_groupanalysis, [('spmT_images', f"l2_analysis_n_{n}.@T"),
            ('con_images', f"l2_analysis_n_{n}.@con")])])

    contrasts = [('Mean effect', 'T', ['mean'], [1])]
    
    # Node for the design matrix
    one_sample_t_test_design = Node(OneSampleTTestDesign(), name = 'one_sample_t_test_design')
    
    if gzip:
        l2_analysis.connect([(sub_contrasts, gunzip, [('contrast_map_sublist', 'in_file')]), 
            (gunzip, one_sample_t_test_design, [("out_file", 'in_files')]),
            (one_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')])])
    else:
        l2_analysis.connect([(sub_contrasts, one_sample_t_test_design, [("contrast_map_sublist", 'in_files')]),
            (one_sample_t_test_design, estimate_model, [('spm_mat_file', 'spm_mat_file')])])

    estimate_contrast.inputs.contrasts = contrasts

    return l2_analysis