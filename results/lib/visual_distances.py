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
from nilearn.image import resample_to_img, binarize_img
from nilearn.masking import intersect_masks
from os.path import join as opj

def plot_2d_multisubject(points, colors, subjects, contrast_list, maps, pipelines,title):
    fig, ax = plt.subplots(figsize=(16, 16), facecolor="white", constrained_layout=True)
    fig.suptitle('truc', size=16)

    x, y = points.T
    col_list = ['green', 'blue', 'red', 'yellow', 'orange', 'pink', 'black', 'purple', 'chocolate', 'navy', 'darkgreen', 'salmon', 'grey']
    annotations = []

    if len(subjects) != 1 and len(contrast_list)==1:
        for img in sorted(maps):
            annotations.append(img.split('/')[-1].split('_')[1].upper())

    elif len(subjects) == 1 and len(contrast_list)!=1:
        for img in sorted(maps):
            annotations.append(img.split('/')[-1].split('_')[3].split('.')[0].upper())
        print(annotations)

    elif len(subjects) != 1 and len(contrast_list)!=1:
        for img in sorted(maps):
            annotations.append(img.split('/')[-1].split('_')[1].upper()+','+img.split('/')[-1].split('_')[3].split('.')[0].upper())
        print(annotations)
        
    for col in np.unique(sorted(colors)):
        index=[i for i in range(len(colors)) if colors[i]==col]
        ax.scatter(x[index],
                   y[index], 
                   #c = col_list[col], 
                   #label=pipelines[col],
                   s=50, alpha=0.8)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
    for i, label in enumerate(colors):
        plt.annotate(label, (x[i], y[i]))
    plt.legend()
    
def resample(img, resolution=2):
    template = load_mni152_template(resolution=resolution)
    res_img = resample_to_img(img, template)

    return res_img

def get_intersection_mask(images_list):
    nii_img_list = []
    for img in images_list:
        nii_img_list.append(binarize_img(resample(nib.load(img),2)))

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

def plot_multisubject_maps(subjects, contrast_list, map_dir, encoder=None):
    assert(len(subjects)<6)
    colors=[]
    data = []
    
    maps = [glob(f'{map_dir}/sub_{sub}_contrast_{con}.nii*') for sub in subjects for con in contrast_list]
    maps = list(chain(*maps))

    pipelines = list(np.unique([img.split('/')[-3] for img in sorted(maps)]))

    if encoder:
        preprocess_type='resampled_masked_normalized'
        training_subset = 'neurovault_dataset'
        model_to_use='4layers'
        training_setup = f"neurovault_dataset_maps_resampled_masked_normalized_{training_subset}_epochs_200" + \
                    f"_batch_size_32_model_cnn_{model_to_use}_lr_1e-04" 
        parameter_file = opj('/Users/egermani/Documents/self_taught_decoding/NeuroVault_dataset', 
                             training_setup, 'model_final.pt')
        model_parameter = torch.load(parameter_file, map_location="cpu")
        model_parameter = model_parameter.eval()

        data = np.zeros((len(list(maps)), 512*3*4*3))
        for i,img in enumerate(sorted(maps)):
            tmp = nib.load(img)
            affine = tmp.affine.copy()
            header = tmp.header

            in_tensor = image_to_tensor(tmp)
            out_tensor = model_parameter.encode(in_tensor)
            data[i]= np.reshape(out_tensor.detach().numpy(), -1)

            img_pipeline = img.split('/')[-3]
            colors.append(pipelines.index(img_pipeline))

    else:
        for i,img in enumerate(sorted(maps)):
            res_img = resample(nib.load(img), 2)
            data.append(np.reshape(np.nan_to_num(res_img.get_fdata()), -1))
            img_pipeline = img.split('/')[-3]
            colors.append(pipelines.index(img_pipeline))

    pca = PCA(n_components=2)
    pca = pca.fit_transform(data)
    title= f"PCA for subject {subjects} and contrasts {contrast_list}"
    #tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=15).fit_transform(np.array(data))
    plot_2d_multisubject(pca, colors, subjects, contrast_list, maps, pipelines,title)




