from glob import glob
from nilearn.image import resample_to_img, resample_img
from nilearn import datasets
import nibabel as nib
import numpy as np
import os.path as op
import os
from nipype.interfaces.fsl import Info

def masking(file):
    # Load mask to apply to images  
    mask = nib.load(Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz'))

    masked_file = os.path.split(file)[0] + '/' + os.path.basename(file).split('.')[0] + '_masked.' + '.'.join(os.path.basename(file).split('.')[1:])

    img = nib.load(file)
    img_data = img.get_fdata()
    img_data = np.nan_to_num(img_data)
    img_affine = img.affine
    
    print("Masking image...")
    
    mask_data = mask.get_fdata()
    
    masked_img_data = img_data * mask_data
    
    masked_img = nib.Nifti1Image(masked_img_data, img_affine)
    
    nib.save(masked_img, masked_file)

    print("Done.") # Save original image resampled and masked

    return masked_file


def rescaling(file, scale_factor):
    affine = nib.load(file).affine
    data = nib.load(file).get_fdata()

    print("Rescaling data...")
    
    scaled_data = data * scale_factor 
    
    img = nib.Nifti1Image(scaled_data, affine)
    
    scaled_file = os.path.split(file)[0] + '/' + os.path.basename(file).split('.')[0] + '_scaled.' + '.'.join(os.path.basename(file).split('.')[1:])

    nib.save(img, scaled_file)

    print("Done.")
    
    return scaled_file
    

def resampling(file):
    img = nib.load(file)
    img_data = img.get_fdata()
    img_data = np.nan_to_num(img_data)
    img_affine = img.affine

    num_img = nib.Nifti1Image(img_data, img_affine)

    res_img = resample_to_img(num_img, Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz'), 
                                      interpolation='nearest', clip = True)

    res_file = os.path.split(file)[0] + '/' + os.path.basename(file).split('.')[0] + '_resampled.' + '.'.join(os.path.basename(file).split('.')[1:])

    nib.save(res_img, res_file)

    return res_file

