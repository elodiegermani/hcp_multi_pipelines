import os 
from glob import glob 
import shutil

def rename_fsl(in_dir, out_dir):
    contrast_dict = {'1':'cue', '2':'left-foot', '3':'left-hand', '4':'right-foot', '5':'right-hand', '6':'tongue'}
    typedict = {'cope':'con', 'tstat':'tstat'}

    in_files = glob(f'{in_dir}/_contrast_*_fwhm_*_hrf_*_nb_param_*_subject_id_*_task_MOTOR/_warpall0/*.nii*')
    print(len(in_files))

    for fi in sorted(in_files):
        c = fi.split('/')[-3].split('_')[2]
        f = fi.split('/')[-3].split('_')[4]
        hrf = fi.split('/')[-3].split('_')[6]
        if hrf == 'no':
            h=0
            p = fi.split('/')[-3].split('_')[10]
            s = fi.split('/')[-3].split('_')[13]
        else:
            h=1
            p = fi.split('/')[-3].split('_')[9]
            s = fi.split('/')[-3].split('_')[12]

        t = fi.split('/')[-1].split('_')[0][:-1]

        contrast = contrast_dict[str(c)]
        typemap = typedict[t]


        new_name = f'sub-{s}_{contrast}_fsl-{f}-{p}-{h}_{typemap}.nii.gz'
        print(new_name)
        
        shutil.copyfile(fi, f'{out_dir}/{new_name}')

def rename_spm(in_dir, out_dir):
    contrast_dict = {'1':'cue', '2':'left-foot', '3':'left-hand', '4':'right-foot', '5':'right-hand', '6':'tongue'}
    typedict = {'con':'con', 'spmT':'tstat'}

    in_files = glob(f'{in_dir}/_fwhm_*_hrf_*_nb_param_*_subject_id_*_task_MOTOR/*.nii*')
    print(len(in_files))

    for fi in in_files:
        c = fi.split('/')[-1].split('.')[0][-1]
        f = fi.split('/')[-2].split('_')[2]

        hrf = fi.split('/')[-2].split('_')[4]
        if hrf == 'no':
            h=0
            p = fi.split('/')[-2].split('_')[8]
            s = fi.split('/')[-2].split('_')[11]
        else:
            h=1
            p = fi.split('/')[-2].split('_')[7]
            s = fi.split('/')[-2].split('_')[10]

        t = fi.split('/')[-1].split('_')[0]

        contrast = contrast_dict[c]
        typemap = typedict[t]


        new_name = f'sub-{s}_{contrast}_spm-{f}-{p}-{h}_{typemap}.nii.gz'
        print(new_name)

        shutil.copyfile(fi, f'{out_dir}/{new_name}')


def rename_group(in_dir, out_dir):
    contrast_dict = {'cue':'cue', 'lf':'left-foot', 'lh':'left-hand', 'rf':'right-foot', 'rh':'right-hand', 
                     't':'tongue'}
    typedict = {'con':'con', 'spmT':'tstat'}

    in_files = glob(f'{in_dir}/DATASET_SOFT_*_FWHM_*_MC_PARAM_*_HRF_*/group_maps/l2_analysis_n_50/_contrast_*/_i_*/*000*.nii')
    print(len(in_files))
    
    for fi in in_files:
        c = fi.split('/')[-3].split('_')[-1]

        f = fi.split('/')[-6].split('_')[4]

        h = fi.split('/')[-6].split('_')[-1]

        p = fi.split('/')[-6].split('_')[7]

        s = fi.split('/')[-2].split('_')[-1]

        t = fi.split('/')[-1].split('_')[0]

        contrast = contrast_dict[c]
        typemap = typedict[t]


        new_name = f'group-{s}_{contrast}_spm-{f}-{p}-{h}_{typemap}.nii.gz'
        print(new_name)

        shutil.copyfile(fi, f'{out_dir}/{new_name}')