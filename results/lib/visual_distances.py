from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from glob import glob
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker
from itertools import chain
from nilearn import datasets
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img, binarize_img, resample_img
from nilearn.masking import intersect_masks, apply_mask
from os.path import join as opj
import pandas as pd
import matplotlib.colors as mcolors
import random


def plot_2d_multisubject(points, colors, label_list, maps, pipelines,title):
    fig, ax = plt.subplots(figsize=(16, 16), facecolor="white", constrained_layout=True)
    fig.suptitle('truc', size=16)

    x, y = points.T
    col_list = list(mcolors.TABLEAU_COLORS.keys())
    #random.shuffle(col_list)
    annotations = []

    for img in sorted(maps):
        annotations.append(img.split('/')[-3].split('_')[2]+','+img.split('/')[-3].split('_')[4]+','+img.split('/')[-3].split('_')[7] +',' + img.split('/')[-3].split('_')[9])
    print(annotations)


        
    for col in np.unique(colors):
        index=[i for i in range(len(colors)) if colors[i]==col]
        ax.scatter(x[index],
                   y[index], 
                   c = col_list[col], 
                   label=label_list[col],
                   s=50, alpha=0.8)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
    for i, label in enumerate(annotations):
        plt.annotate(annotations[i], (x[i], y[i]))
    plt.legend()
    
def resample(img, resolution=2):
    template = load_mni152_template(resolution=resolution)
    res_img = resample_to_img(img, template)

    return res_img

def get_intersection_mask(images_list):
    nii_img_list = []
    for img in images_list:
        nii_img_list.append(binarize_img(img))

    mask = intersect_masks(nii_img_list)

    return mask

def image_to_tensor(img):
    '''
    Convert Nifti1Image or load filename and return tensor to be input to the model.

    Parameters:
        - img, str or Nifti1Image

    Return:
        - sample, FloatTensor
    '''

    if isinstance(img, nib.Nifti1Image):
        sample = img
    else:
        sample = nib.load(img)

    sample = sample.get_data().copy().astype(float)
    sample = np.nan_to_num(sample)

    sample = torch.tensor(sample).view((1), (1), *sample.shape)
    
    return sample.type('torch.FloatTensor')

def plot_multisubject_maps(subjects, contrast_list, map_dir, encoder=None, plot=True, cluster=True, n_cluster=2):
    colors=[]
    data = []
    
    maps = [glob(f'{map_dir}/sub_{sub}_contrast_{con}.nii*') for sub in subjects for con in contrast_list]
    maps = list(chain(*maps))
    maps = sorted(maps)

    pipelines = list(np.unique([img.split('/')[-3] for img in sorted(maps)]))

    if encoder:
        preprocess_type='resampled_masked_normalized'
        training_subset = 'neurovault_dataset'
        model_to_use='5layers'
        training_setup = f"neurovault_dataset_maps_resampled_masked_normalized_{training_subset}_epochs_200" + \
                    f"_batch_size_32_model_cnn_{model_to_use}_lr_1e-04" 
        parameter_file = opj('/Users/egermani/Documents/self_taught_decoding/NeuroVault_dataset', 
                             training_setup, 'model_final.pt')
        model_parameter = torch.load(parameter_file, map_location="cpu")
        model_parameter = model_parameter.eval()

        data = np.zeros((len(list(maps)), 512*2*2*2))
        for i,img in enumerate(sorted(maps)):
            tmp = nib.load(img)
            affine = tmp.affine.copy()
            header = tmp.header

            affine[:3,:3] = np.sign(affine[:3,:3]) * 4
            shape = (48, 56, 48)

            tmp = resample_img(tmp, target_affine=affine, target_shape=shape, interpolation='nearest')

            in_tensor = image_to_tensor(tmp)
            out_tensor = model_parameter.encode(in_tensor)
            data[i]= np.reshape(out_tensor.detach().numpy(), -1)

            img_pipeline = img.split('/')[-3]
            img_subject = img.split('/')[-1].split('_')[1]
            img_contrast = img.split('/')[-1].split('_')[3].split('.')[0]
            if len(subjects) > 1 and len(contrast_list) == 1:
                colors.append(subjects.index(img_subject))
                label_list = subjects
            elif len(subjects) == 1 and len(contrast_list) > 1:
                colors.append(contrast_list.index(img_contrast))
                label_list = contrast_list
            else:
                colors.append(0)
                label_list = subjects

    else:
        image_list = []
        for i,f in enumerate(sorted(maps)):
            img = nib.load(f)
            res_img = resample(img)
            image_list.append(res_img)

        mask = get_intersection_mask(image_list)

        for img, f in zip(image_list, sorted(maps)):
            data.append(np.reshape(np.nan_to_num(img.get_fdata() * mask.get_fdata()), -1))
            img_pipeline = f.split('/')[-3]
            img_subject = f.split('/')[-1].split('_')[1]
            img_contrast = f.split('/')[-1].split('_')[3].split('.')[0]
            if len(subjects) > 1 and len(contrast_list) == 1:
                colors.append(subjects.index(img_subject))
                label_list = subjects
            elif len(subjects) == 1 and len(contrast_list) > 1:
                colors.append(contrast_list.index(img_contrast))
                label_list = contrast_list
            else:
                colors.append(0)
                label_list = subjects

    pca = PCA(n_components=2)
    pca = pca.fit_transform(data)

    title= f"PCA for subject {subjects} and contrasts {contrast_list}"
    if plot:
        plot_2d_multisubject(pca, colors, label_list, maps, pipelines,title)

    if cluster:
        cluster_pca(pca, maps, n_cluster)

    return pca


def plot_group_maps(contrast_list, map_dir, encoder=None, plot=True, cluster=True, n_cluster=2):
    colors=[]
    data = []
    model = None
    
    maps = [glob(f'{map_dir}/contrast_{con}.nii*') for con in contrast_list]
    maps = list(chain(*maps))
    maps = sorted(maps)

    pipelines = list(np.unique([img.split('/')[-3] for img in sorted(maps)]))

    if encoder:
        preprocess_type='resampled_masked_normalized'
        training_subset = 'neurovault_dataset'
        model_to_use='5layers'
        training_setup = f"neurovault_dataset_maps_resampled_masked_normalized_{training_subset}_epochs_200" + \
                    f"_batch_size_32_model_cnn_{model_to_use}_lr_1e-04" 
        parameter_file = opj('/Users/egermani/Documents/self_taught_decoding/NeuroVault_dataset', 
                             training_setup, 'model_final.pt')
        model_parameter = torch.load(parameter_file, map_location="cpu")
        model_parameter = model_parameter.eval()

        data = np.zeros((len(list(maps)), 512*2*2*2))
        for i,img in enumerate(sorted(maps)):
            tmp = nib.load(img)
            affine = tmp.affine.copy()
            header = tmp.header

            affine[:3,:3] = np.sign(affine[:3,:3]) * 4
            shape = (48, 56, 48)

            tmp = resample_img(tmp, target_affine=affine, target_shape=shape, interpolation='nearest')

            in_tensor = image_to_tensor(tmp)
            out_tensor = model_parameter.encode(in_tensor)
            data[i]= np.reshape(out_tensor.detach().numpy(), -1)

            img_pipeline = img.split('/')[-3]
            img_contrast = img.split('/')[-1].split('_')[1].split('.')[0]
            if len(contrast_list) > 1:
                colors.append(contrast_list.index(img_contrast))
                label_list = contrast_list
            else:
                colors.append(0)
                label_list = contrast_list

    else:
        image_list = []
        for i,f in enumerate(sorted(maps)):
            img = nib.load(f)
            res_img = resample(img)
            image_list.append(res_img)

        mask = get_intersection_mask(image_list)

        for img, f in zip(image_list, sorted(maps)):
            data.append(np.reshape(np.nan_to_num(img.get_fdata() * mask.get_fdata()), -1))
            img_pipeline = f.split('/')[-3]
            img_contrast = f.split('/')[-1].split('_')[1].split('.')[0]
            if len(contrast_list) > 1:
                colors.append(contrast_list.index(img_contrast))
                label_list = contrast_list
            else:
                colors.append(0)
                label_list = contrast_list

    pca = PCA(n_components=2)
    pca = pca.fit_transform(data)

    title= f"PCA for group maps of contrasts {contrast_list}"
    if plot:
        plot_2d_multisubject(pca, colors, label_list, maps, pipelines,title)

    if cluster:
        model = cluster_pca(pca, maps, n_cluster)


    return pca, model, maps

def find_n_clusters(pca):
    pca_dataframe = pd.DataFrame(pca)
    wcss=[]
    for k in range(1, len(pca)):
        
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)
        
        # Fit model to samples
        model.fit(pca_dataframe)
        wcss.append(model.inertia_)

    plt.figure(figsize=(10, 8))
    plt.plot(range(1,len(pca)), wcss, marker='o', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('K-means with PCA Clustering')
    plt.show()

def cluster_pca(pca, maps, n_clusters):
    col_list = list(mcolors.TABLEAU_COLORS.keys())
    #random.shuffle(col_list)
    pca_dataframe = pd.DataFrame(pca[:, 0:2], columns=['Comp 1', 'Comp 2'])
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=n_clusters)
    
    # Fit model to samples
    model.fit(pca_dataframe)

    pca_dataframe['K-means clusters']=model.labels_

    x,y=pca[:, 0:2].T

    annotations = []

    for img in sorted(maps):
        annotations.append(img.split('/')[-3].split('_')[2]+','+img.split('/')[-3].split('_')[4]+','+img.split('/')[-3].split('_')[7] +',' + img.split('/')[-3].split('_')[9])
    
    
    f = plt.figure(figsize=(16, 16))
    for c in np.unique(pca_dataframe['K-means clusters'].tolist()):
        plt.scatter(pca_dataframe['Comp 1'][pca_dataframe['K-means clusters']==c], 
            pca_dataframe['Comp 2'][pca_dataframe['K-means clusters']==c], 
            c= col_list[c],
            label = f'Cluster {c}')

    for i, label in enumerate(annotations):
        plt.annotate(annotations[i], (x[i], y[i]))
    plt.legend()

    centers = model.cluster_centers_

    for c in range(len(centers)):
        plt.scatter(centers[c][0], centers[c][1], s=200, c=col_list[c], marker='s')
    plt.title('Clustering of group statistic maps on PCA components.')

    plt.show()
    mode = maps[0].split('/')[-4]
    f.savefig(f'../figures/clustering_{mode}_maps.png')

    return model




    






