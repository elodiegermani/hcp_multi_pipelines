import numpy as np 
import nibabel as nib 
from nilearn import image, plotting, datasets, masking, glm 
from nibabel.processing import resample_from_to
from scipy import stats
import pandas as pd 
from os.path import join as opj
import matplotlib.pyplot as plt
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps


def mask_using_intersect(img1, img2):
    '''
    Compute the mask of the original map and apply it to the reconstructed one. 
    '''
    img1_bin = image.binarize_img(img1)
    img2_bin = image.binarize_img(img2)
    
    global_mask = masking.intersect_masks([img1_bin, img2_bin], 1)
    
    img1_masked_data = img1.get_fdata() * global_mask.get_fdata()
    img2_masked_data = img2.get_fdata() * global_mask.get_fdata()
    
    img1_masked = nib.Nifti1Image(img1_masked_data, img1.affine)
    img2_masked = nib.Nifti1Image(img2_masked_data, img2.affine)

    return img1_masked, img2_masked

def mask_using_original(img1, mask):
    masked_data = img1.get_fdata() * mask.get_fdata()
    
    masked_img = nib.Nifti1Image(masked_data, img1.affine)
    
    return masked_img

def t_to_z(t_stat_img, N):
    # Convert t-statistic images to z-statistic images used for the consensus analysis
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

def get_percent_overlap(img1, img2, roi):
    roi_data = roi.get_fdata() > 1e-6
    data1 = np.nan_to_num(img1.get_fdata() * roi_data.astype('int'))
    data2 = np.nan_to_num(img2.get_fdata() * roi_data.astype('int'))

    img_rec = nib.Nifti1Image(data1, img1.affine)
    img2_rec = nib.Nifti1Image(data2, img2.affine)

    # Vectorise input data
    data1 = np.reshape(data1, -1)
    data2 = np.reshape(data2, -1)
    
    data1 = np.nan_to_num(data1)
    data2 = np.nan_to_num(data2)

    data_overlap = data1 * data2 
    data_overlap = data_overlap > 0
    
    percent_overlap = np.count_nonzero(data_overlap)/np.count_nonzero(data1 > 0)
    print(np.count_nonzero(data1), np.count_nonzero(data2))
    
    return percent_overlap

def get_percent_activated(img1, roi):
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
    atlas = '/srv/tempdd/egermani/hcp_pipelines/results/atlas'
    atlas_dict_unthresh = {'cue':'tfMRI_MOTOR_CUE-AVG_zstat1.nii.gz', 'lf':'tfMRI_MOTOR_LF-AVG_zstat1.nii.gz', 
                       'lh':'tfMRI_MOTOR_LH-AVG_zstat1.nii.gz', 'rf':'tfMRI_MOTOR_RF-AVG_zstat1.nii.gz', 'rh':'tfMRI_MOTOR_RH-AVG_zstat1.nii.gz', 
                       't':'tfMRI_MOTOR_T-AVG_zstat1.nii.gz'}
                       
    df = pd.DataFrame(columns=['name','contrast','percent_overlap'])
    m = len(stat_maps)
    atlas_roi = datasets.fetch_atlas_juelich('prob-2mm')
    lab =[31,32]
    masks = [image.index_img(atlas_roi.maps, lab_i) for lab_i in lab]
    mask, mask_right, mask_left = compute_unilateral_masks(masks)

    

    for i, f in enumerate(stat_maps):
        print(f"Map {i} on {m}")
        contrast = f.split('/')[-1].split('_')[-1].split('.')[0]
        print(contrast)
        f_name = f.split('/')[-1].split('.')[0]

        if contrast == 'lf' or contrast == 'lh':
            roi_mask = mask_left 
        elif contrast =='rf' or contrast == 'rh':
            roi_mask = mask_right
        else:
            roi_mask = mask

        #atlas_unthresh = nib.load(opj(atlas, atlas_dict_unthresh[contrast]))
        
        mask_unthresh_t = image.binarize_img(nib.load(f))
        unthresh_t = nib.load(f)

        resampled_unthresh_t = resample_from_to(unthresh_t, roi_mask)
        resampled_mask_unthresh_t = resample_from_to(mask_unthresh_t, roi_mask)

        masked_unthresh_t = mask_using_original(resampled_unthresh_t, resampled_mask_unthresh_t)
                
        unthresh_z = t_to_z(masked_unthresh_t, 50)


        thresh_z, threshold = glm.threshold_stats_img(unthresh_z, alpha=0.05, height_control='fdr',
                                                     two_sided=False)
        '''
        plotting.plot_glass_brain(unthresh_z, colorbar = True, annotate=False, plot_abs=False, threshold=None, cmap=nilearn_cmaps['cold_hot'],title='Our unthresholded map')
        plotting.plot_glass_brain(thresh_z, title='Our thresholded map')
        plotting.plot_glass_brain(mask_atlas_unthresh, colorbar = True, annotate=False, plot_abs=False, threshold=None, cmap=nilearn_cmaps['cold_hot'], title='NeuroQuery thresholded')
        plotting.plot_glass_brain(atlas_thresh, title='NeuroQuery thresholded')
        plt.show()
        '''
        percent_activated = get_percent_activated(thresh_z, roi_mask)
        #percent_overlap = get_percent_overlap(atlas_thresh, thresh_z, roi)
        #print(percent_overlap)

        df_file = pd.DataFrame([[f_name,contrast,percent_activated]], columns=['name','contrast','percent_activated'])

        df = pd.concat([df, df_file], ignore_index=True)

    return df 