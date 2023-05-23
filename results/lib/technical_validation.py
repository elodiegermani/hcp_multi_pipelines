import numpy as np 
import nibabel as nib 
from nilearn import image, plotting, datasets, masking, glm 
from nibabel.processing import resample_from_to
from scipy import stats
import pandas as pd 
from os.path import join as opj


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

def get_percent_overlap(img1, img2):
    # Vectorise input data
    data1 = np.reshape(img1.get_fdata(), -1)
    data2 = np.reshape(img2.get_fdata(), -1)

    data_overlap = data1 * data2 
    data_overlap = data_overlap > 0
    
    percent_overlap = sum(data_overlap)/sum(data1 > 0)
    
    return percent_overlap

def run_technical_validation(maps):
    atlas = '/srv/tempdd/egermani/hcp_pipelines/results/atlas'
    atlas_dict_unthresh = {'cue':'movement.nii.gz', 'lf':'left_foot.nii.gz', 
                       'lh':'left_hand.nii.gz', 'rf':'right_foot.nii.gz', 'rh':'right_hand.nii.gz', 
                       't':'tongue.nii.gz'}
                       
    df = pd.DataFrame(columns=['name','contrast','percent_overlap'])
    m = len(maps)

    for i, f in enumerate(maps):
        print(f"Map {i} on {m}")
        contrast = f.split('/')[-1].split('_')[-1].split('.')[0]
        f_name = f.split('/')[-1].split('.')[0]
        
        atlas_unthresh = nib.load(opj(atlas, atlas_dict_unthresh[contrast]))
        
        mask_unthresh_t = image.binarize_img(nib.load(f))
        
        unthresh_t = resample_from_to(nib.load(f), atlas_unthresh)
        res_mask_unthresh_t = resample_from_to(mask_unthresh_t, atlas_unthresh)
        
        unthresh_t = mask_using_original(unthresh_t, res_mask_unthresh_t)
        
        mask_unthresh_t, mask_atlas_unthresh = mask_using_intersect(unthresh_t, atlas_unthresh)
        
        atlas_thresh, threshold = glm.threshold_stats_img(mask_atlas_unthresh, alpha=0.05, height_control='fdr', 
                                                          two_sided=False)
        unthresh_z = t_to_z(mask_unthresh_t, 50)


        thresh_z, threshold = glm.threshold_stats_img(unthresh_z, alpha=0.05, height_control='fdr',
                                                     two_sided=False)
        
        
        percent_overlap = get_percent_overlap(atlas_thresh, thresh_z)
        print(percent_overlap)

        df_file = pd.DataFrame([[f_name,contrast,percent_overlap]], columns=['name','contrast','percent_overlap'])

        df = pd.concat([df, df_file], ignore_index=True)

    return df 