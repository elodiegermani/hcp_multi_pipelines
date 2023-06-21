import numpy as np 
import nibabel as nib 
from nilearn import image, plotting, datasets, masking, glm 
from nibabel.processing import resample_from_to
from scipy import stats
import pandas as pd 
from os.path import join as opj
import matplotlib.pyplot as plt
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps

def mask_using_original(img1, mask):
    '''
    Mask an image using a precomputed mask. 

    Parameters:
        - img1, Nifti1Image: image to mask
        - mask, Nifti1Image: mask to apply

    Returns:
        - masked_img, Nifti1Image: masked image
    '''
    masked_data = img1.get_fdata() * mask.get_fdata()
    
    masked_img = nib.Nifti1Image(masked_data, img1.affine)
    
    return masked_img

def t_to_z(t_stat_img, N):
    '''
    Convert t-statistic maps in Z-statistic map.

    Parameters:
        - t_stat_img, Nifti1Image: image to convert
        - N, int: degrees of freedom (sample size - 1)

    Returns:
        - z_stat_img, Nifti1Image: z_statistic image
    '''

    df = N-1

    t_stat = np.nan_to_num(t_stat_img.get_fdata())
    z_stat = np.zeros_like(t_stat)

    # Handle large and small values differently to avoid underflow
    z_stat[t_stat < 0] = stats.norm.ppf(stats.t.sf(-t_stat[t_stat < 0], df))
    z_stat[t_stat > 0] = -stats.norm.ppf(stats.t.sf(t_stat[t_stat > 0], df))

    #z_stat[t_stat < 0] = -stats.norm.ppf(stats.t.cdf(-t_stat[t_stat < 0], df))
    #z_stat[t_stat > 0] = stats.norm.ppf(stats.t.cdf(t_stat[t_stat > 0], df))
    
    z_stat = np.nan_to_num(z_stat)
    
    z_stat_img = nib.Nifti1Image(z_stat, t_stat_img.affine)
        
    return(z_stat_img)

def get_percent_activated(img1, roi):
    '''
    Compute the percent of activated voxels inside a ROI.

    Parameters:
        - img1, Nifti1Image: statistic map in which we want to compute the value
        - roi, Nifti1Image: atlas of the ROI of interest

    Returns:
        - percent_activated, float: percentage of activated voxels inside the ROI for the input image.
    '''
    roi_data = roi.get_fdata() > 1e-6
    data1 = np.nan_to_num(img1.get_fdata() * roi_data.astype('int'))

    img_rec = nib.Nifti1Image(data1, img1.affine)
    roi_rec = nib.Nifti1Image(roi_data, roi.affine)
    plotting.plot_glass_brain(img_rec)
    plotting.plot_glass_brain(roi_rec)
    plt.show()

    # Vectorise input data
    data1 = np.reshape(data1, -1)
    data1 = np.nan_to_num(data1)
    
    percent_activated = np.count_nonzero(data1)/np.count_nonzero(roi_data)
    print(np.count_nonzero(data1), np.count_nonzero(roi_data))
    
    return percent_activated

def compute_unilateral_masks(masks):
    '''
    Compute the global mask of 2 ROI and specific mask for each hemisfer. 

    Parameters:
        - masks, list of Nifti1Image: masks of the two ROI

    Returns:
        - mask, Nifti1Image: global mask
        - mask_right, Nifti1Image: right hemisfer mask
        - mask_left, Nifti1Image: left hemisfer mask
    '''
    mask_data=masks[0].get_fdata() + masks[1].get_fdata()

    mask = image.new_img_like(masks[0], mask_data, affine=masks[0].affine, copy_header=True)

    x_dim = mask_data.shape[0]
    x_center = int(x_dim/2)

    mask_data_left = mask_data.copy()
    mask_data_left[0:x_center,:,:] = 0
    mask_left = image.new_img_like(masks[0], mask_data_left, affine=masks[0].affine, copy_header=True)

    mask_data_right = mask_data.copy()
    mask_data_right[x_center:,:,:] = 0
    mask_right = image.new_img_like(masks[0], mask_data_right, affine=masks[0].affine, copy_header=True)

    return mask, mask_right, mask_left

def run_technical_validation(stat_maps):     
    '''
    Perform technical validation. 
    '''         
    df = pd.DataFrame(columns=['name','contrast','percent_overlap'])
    m = len(stat_maps)
    atlas_roi = datasets.fetch_atlas_juelich('prob-2mm')
    lab =[31,32]
    masks = [image.index_img(atlas_roi.maps, lab_i) for lab_i in lab]
    mask, mask_right, mask_left = compute_unilateral_masks(masks)
    

    for i, f in enumerate(stat_maps):
        print(f"Map {i} on {m}")
        contrast = f.split('/')[-1].split('_')[1]
        print(contrast)
        f_name = f.split('/')[-1].split('.')[0]

        if contrast == 'left-foot' or contrast == 'left-hand':
            roi_mask = mask_right # Controlateral activation
        elif contrast =='right-foot' or contrast == 'right-hand':
            roi_mask = mask_left 
        else:
            roi_mask = mask
        
        mask_unthresh_t = image.binarize_img(nib.load(f))
        unthresh_t = nib.load(f)

        resampled_unthresh_t = resample_from_to(unthresh_t, roi_mask)
        resampled_mask_unthresh_t = resample_from_to(mask_unthresh_t, roi_mask) 
        masked_unthresh_t = mask_using_original(resampled_unthresh_t, resampled_mask_unthresh_t) # Resample each map to the same dimensiions 
        # and apply mask to remove artifacts of resampling
                
        unthresh_z = t_to_z(masked_unthresh_t, 50)

        thresh_z, threshold = glm.threshold_stats_img(unthresh_z, alpha=0.05, height_control='fdr',
                                                     two_sided=False)
        percent_activated = get_percent_activated(thresh_z, roi_mask)

        df_file = pd.DataFrame([[f_name,contrast,percent_activated]], columns=['name','contrast','percent_activated'])

        df = pd.concat([df, df_file], ignore_index=True)

    return df 