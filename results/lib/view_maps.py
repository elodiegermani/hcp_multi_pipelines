import matplotlib.pyplot as plt
from nilearn import plotting
from glob import glob 
import os
from math import * 
import matplotlib
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img, binarize_img, resample_img
from nilearn.masking import intersect_masks, apply_mask, compute_background_mask
import nibabel as nib
import numpy as np
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
from os.path import join as opj


def resample(img, resolution=4):
    '''
    Resample images to a MNI152 template.
    '''
    template = load_mni152_template(resolution=resolution)
    res_img = resample_to_img(img, template)

    return res_img

def get_intersection_mask(images_list):
    '''
    Compute intersection mask given a list of images.
    '''
    nii_img_list = []
    for img in images_list:
        nii_img_list.append(binarize_img(img, threshold=1e-6))

    mask = intersect_masks(nii_img_list, threshold=1)

    return mask

def visualise_subject_maps(subject, contrast, data_dir):
    '''
    Plot the statistic maps obtained with the different pipelines for a given subject.
    
    Parameters:
        - subject, str: ID of the subject
        - contrast, str: contrast chosen
        - data_dir, str: path to all data
    '''
    maps = glob(f'{data_dir}/sub-{subject}_{contrast}_*_tstat.nii*')
    
    pipelines = [img.split('/')[-1].split('_')[2].split('-')[0].upper() +','+img.split('/')[-1].split('_')[2].split('-')[1]+','+img.split('/')[-1].split('_')[2].split('-')[2] +',' + img.split('/')[-1].split('_')[2].split('-')[3] for img in sorted(maps)]
    
    f = plt.figure(figsize = (7 * 6, 7 * 4))
    f.suptitle(f'Subject {subject}', backgroundcolor = 'black', color='white', fontsize=30, fontweight='bold')
    gs = f.add_gridspec(4, 6)
    
    image_list = []
    for i,file in enumerate(sorted(maps)):
        img = nib.load(file)
        res_img = resample(img)
        image_list.append(res_img)

    mask = get_intersection_mask(image_list)
    
    for i, img in enumerate(image_list):
        img_data = img.get_fdata() * mask.get_fdata()
        img_masked = nib.Nifti1Image(img_data, img.affine)
        ax = f.add_subplot(gs[floor(i/6), i - 6 * floor(i/6)])
        
        disp = plotting.plot_glass_brain(img_masked, figure=f, axes=ax, 
                               display_mode = 'z', colorbar = False, annotate=False, 
                                         cmap=nilearn_cmaps['cold_hot'], plot_abs=False)
        disp.title(pipelines[i], size=28)
        
    return f

def visualise_group_maps(group, contrast, data_dir):
    '''
    Plot the statistic maps obtained with the different pipelines for a given group.
    
    Parameters:
        - group, str: ID of the group
        - contrast, str: contrast chosen
        - data_dir, str: path to all data
    '''
    maps = glob(f'{data_dir}/group-{group}_{contrast}_*_tstat.nii')

    pipelines = [img.split('/')[-1].split('_')[2].split('-')[0].upper() +','+img.split('/')[-1].split('_')[2].split('-')[1]+','+img.split('/')[-1].split('_')[2].split('-')[2] +',' + img.split('/')[-1].split('_')[2].split('-')[3] for img in sorted(maps)]
    
    f = plt.figure(figsize = (7 * 6, 7 * 4))
    gs = f.add_gridspec(4, 6)
    f.suptitle(f'Group {group}', backgroundcolor = 'black', color='white', fontsize=30, fontweight='bold')
    
    image_list = []
    for i,file in enumerate(sorted(maps)):
        img = nib.load(file)
        res_img = resample(img)
        image_list.append(res_img)

    mask = get_intersection_mask(image_list)
    
    for i, img in enumerate(image_list):
        img_data = img.get_fdata() * mask.get_fdata()
        img_masked = nib.Nifti1Image(img_data, img.affine)
        ax = f.add_subplot(gs[floor(i/6), i - 6 * floor(i/6)])
        
        disp = plotting.plot_glass_brain(img_masked, figure=f, axes=ax, 
                               display_mode = 'z', colorbar = False, annotate=False, 
                                         cmap=nilearn_cmaps['cold_hot'], plot_abs=False)
        disp.title(pipelines[i], size=28)
        
    return f