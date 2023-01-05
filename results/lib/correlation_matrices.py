import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import scipy
import matplotlib.gridspec
from nilearn.image import resample_to_img, binarize_img
from nilearn.masking import intersect_masks, apply_mask
from nilearn.datasets import load_mni152_template
import glob
import os

def get_correlation(orig, repro, mask):
    '''
    Compute the Pearson's correlation coefficient between original and reconstructed images.
    '''
    data1 = orig.get_fdata().copy()
    data2 = repro.get_fdata().copy()
    mask_data = mask.get_fdata().copy()

    
    # Vectorise input data
    data1 = np.reshape(data1, -1)
    data2 = np.reshape(data2, -1)
    mask_data = np.reshape(mask_data, -1)

    in_mask_indices = np.logical_not(
            np.logical_or(np.isnan(mask_data), np.absolute(mask_data) == 0))

    data1 = data1[in_mask_indices]
    data2 = data2[in_mask_indices]
    
    corr_coeff = np.corrcoef(data1, data2)[0][1]
    
    return corr_coeff

def resample(img, resolution=2):
    template = load_mni152_template(resolution=resolution)
    res_img = resample_to_img(img, template)

    return res_img

def get_intersection_mask(images_list):
    nii_img_list = []
    for img in images_list:
        nii_img_list.append(binarize_img(img, threshold=1e-6))

    mask = intersect_masks(nii_img_list)

    return mask


def get_correlation_matrix_subject(contrast_list, subject_list, data_dir, software='all'):
    if software == 'all':
        p_global = sorted(glob.glob(f'{data_dir}/*/original/*'))
    else:
        p_global = sorted(glob.glob(f'{data_dir}/*{software}*/original/*'))

    p = [img for img in sorted(p_global) if os.path.basename(img).split('_')[1] in subject_list and os.path.basename(img).split('_')[3].split('.')[0] in contrast_list]
    pipelines = [img.split('/')[-3] for img in sorted(p)]

    target_corr = np.zeros((len(p), len(p)))

    image_list = []

    for f in sorted(p):
        img = nib.load(f)
        res_img = resample(img)
        image_list.append(res_img)

    mask = get_intersection_mask(image_list)

    for i, img1 in enumerate(image_list):
        for j, img2 in enumerate(image_list):   
            target_corr[i,j] = get_correlation(img1, img2, mask)

    cc_df = pd.DataFrame(target_corr, index=pipelines, columns=pipelines)

    return cc_df

def get_correlation_matrix_group(contrast_list, data_dir, software='all'):
    if software == 'all':
        p_global = sorted(glob.glob(f'{data_dir}/*/original/*'))
    else:
        p_global = sorted(glob.glob(f'{data_dir}/*{software}*/original/*'))

    p = [img for img in sorted(p_global) if os.path.basename(img).split('_')[-1].split('.')[0] in contrast_list]
    pipelines = [img.split('/')[-3] for img in sorted(p)]

    target_corr = np.zeros((len(p), len(p)))

    image_list = []

    for f in sorted(p):
        img = nib.load(f)
        res_img = resample(img)
        image_list.append(res_img)

    mask = get_intersection_mask(image_list)

    for i, img1 in enumerate(image_list):
        for j, img2 in enumerate(image_list):   
            target_corr[i,j] = get_correlation(img1, img2, mask)

    cc_df = pd.DataFrame(target_corr, index=pipelines, columns=pipelines)

    return cc_df

def plot_hierarchical_clustering_subject(contrast_list, subject_list, data_dir, software='all'):
    cc_df = get_correlation_matrix(contrast_list, subject_list, data_dir, software)
    n_sub = len(subject_list)
    n_con = len(contrast_list)

    colors = []
    for name in cc_df.columns:
        if 'FSL' in name:
            colors.append('green')
        else:
            colors.append('blue')

    cm = seaborn.clustermap(cc_df,cmap='vlag',method='ward',vmin=-1,vmax=1, figsize=(16, 10), xticklabels=False)
    cm.fig.suptitle(f'SUBJECT {subject_list[0]}, CONTRAST {contrast_list[0].upper()}', 
        size=24, fontweight='bold') 

    #cm.ax_heatmap.tick_params(left=True, bottom=False)

    #cm.ax_heatmap.set_xticklabels(cm.ax_heatmap.get_xmajorticklabels(), fontsize = 16, fontweight='bold')
    cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.get_ymajorticklabels(), fontsize = 16, fontweight='bold')

    #for i, ticklabel in enumerate(cm.ax_heatmap.xaxis.get_majorticklabels()):
    #    if 'FSL' in ticklabel.get_text():
    #        ticklabel.set_color('green')
    #    else:
    #        ticklabel.set_color('blue')

    for i, ticklabel in enumerate(cm.ax_heatmap.yaxis.get_majorticklabels()):
        if 'FSL' in ticklabel.get_text():
            ticklabel.set_color('green')
        else:
            ticklabel.set_color('blue')


    plt.show()
    if len(subject_list)==1 and len(contrast_list) > 1:
        cm.savefig(f'../figures/heatmap_subject_{subject_list}_{n_con}_contrasts.png')
    elif len(subject_list) >=1 and len(contrast_list) == 1:
        cm.savefig(f'../figures/heatmap_{subject_list}_subject_contrasts_{contrast_list[0]}.png')
    elif len(subject_list)==1 and len(contrast_list)==1:
        cm.savefig(f'../figures/heatmap_subject_{subject_list}_contrasts_{contrast_list[0]}.png')
    else:
        cm.savefig(f'../figures/heatmap_{subject_list}_subject_{contrast_list[0]}_contrasts.png')

def plot_hierarchical_clustering_group(contrast_list, data_dir, software='all'):
    cc_df = get_correlation_matrix_group(contrast_list, data_dir, software)
    n_con = len(contrast_list)

    colors = []
    for name in cc_df.columns:
        if 'FSL' in name:
            colors.append('green')
        else:
            colors.append('blue')

    cm = seaborn.clustermap(cc_df,cmap='vlag',method='ward',vmin=-1,vmax=1, figsize=(16, 10), xticklabels=False)
    cm.fig.suptitle(f'CONTRAST {contrast_list[0].upper()}', 
        size=24, fontweight='bold') 

    #cm.ax_heatmap.tick_params(left=True, bottom=False)

    #cm.ax_heatmap.set_xticklabels(cm.ax_heatmap.get_xmajorticklabels(), fontsize = 16, fontweight='bold')
    cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.get_ymajorticklabels(), fontsize = 16, fontweight='bold')

    #for i, ticklabel in enumerate(cm.ax_heatmap.xaxis.get_majorticklabels()):
    #    if 'FSL' in ticklabel.get_text():
    #        ticklabel.set_color('green')
    #    else:
    #        ticklabel.set_color('blue')

    for i, ticklabel in enumerate(cm.ax_heatmap.yaxis.get_majorticklabels()):
        if 'FSL' in ticklabel.get_text():
            ticklabel.set_color('green')
        else:
            ticklabel.set_color('blue')

    mode = data_dir.split('/')[-1]
    cm.savefig(f'../figures/heatmap_group_level_{mode}_{contrast_list[0]}_contrasts.png')
    plt.show()

