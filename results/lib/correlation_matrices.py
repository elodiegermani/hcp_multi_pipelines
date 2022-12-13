import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import scipy
import matplotlib.gridspec
from nilearn.image import resample_to_img, binarize_img
from nilearn.masking import intersect_masks
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

def get_intersection_mask(images_list):
    nii_img_list = []
    for img in images_list:
        nii_img_list.append(binarize_img(nib.load(img)))

    mask = intersect_masks(nii_img_list)

    return mask


def get_correlation_matrix(contrast_list, subject_list, data_dir, software='all'):
    if software == 'all':
        p_global = glob.glob(f'{data_dir}/*/original/*')
    else:
        p_global = glob.glob(f'{data_dir}/*{software}*/original/*')

    p = [img for img in sorted(p_global) if os.path.basename(img).split('_')[1] in subject_list and os.path.basename(img).split('_')[3].split('.')[0] in contrast_list]
    pipelines = [img.split('/')[-1][:-4] + ',' + img.split('/')[-3] for img in p]

    target_corr = np.zeros((len(p), len(p)))

    mask = get_intersection_mask(p)

    for i, p1 in enumerate(p):
        for j, p2 in enumerate(p):   
            img1 = nib.load(p1)
            img2 =  nib.load(p2)
            
            img2 = resample_to_img(img2, img1)
            target_corr[i,j] = get_correlation(img1, img2, mask)

    cc_df = pd.DataFrame(target_corr, index=sorted(pipelines), columns=sorted(pipelines))

    return cc_df

def plot_hierarchical_clustering(contrast_list, subject_list, data_dir, software='all'):
    cc_df = get_correlation_matrix(contrast_list, subject_list, data_dir, software)
    n_sub = len(subject_list)
    n_con = len(contrast_list)
    cm = seaborn.clustermap(cc_df,cmap='vlag',method='ward',vmin=-1,vmax=1, figsize=(16, 16))
    cm.fig.suptitle(f'Correlation matrix for subjects {subject_list} and contrasts {contrast_list}') 
    plt.show()
    if len(subject_list)==1 and len(contrast_list) > 1:
        plt.savefig(f'../figures/heatmap_subject_{subject_list[0]}_{n_con}_contrasts.png')
    elif len(subject_list) >=1 and len(contrast_list) == 1:
        plt.savefig(f'../figures/heatmap_{n_sub}_subject_contrasts_{contrast_list[0]}.png')
    elif len(subject_list)==1 and len(contrast_list)==1:
        plt.savefig(f'../figures/heatmap_subject_{subject_list[0]}_contrasts_{contrast_list[0]}.png')
    else:
        plt.savefig(f'../figures/heatmap_{n_sub}_subject_{n_con}_contrasts.png')

