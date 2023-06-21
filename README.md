# HCP PIPELINES 

This repository contains pipelines used to analyse HCP fMRI data with subject-level analytic pipelines using FSL and SPM and different parameters. It also contains scripts to perform group-level analysis with one sample t-test (within group). 

## Table of contents
   * [How to cite?](#how-to-cite)
   * [Contents overview](#contents-overview)
   * [Installing environment](#installing-environment)
   * [Download necessary data](#download-necessary-data)
   * [Reproducing subject-level analyses](#reproducing-subject-level-analyses)
   * [Reproducing group-level analyses](#reproducing-group-level-analyses)

## How to cite?


## Contents overview

### `src`

This directory contains scripts and notebooks used to launch the analysis of raw data with pipelines. 

### `data`

This directory is made to contain data that will be used by scripts/notebooks stored in the `src` directory and to contain results of those scripts. For details, check [here](#download-necessary-data).

### `results`

This directory contains notebooks and scripts that were used to analyze the results of the experiments. These notebooks were used to evaluate the quality of data obtained from different pipelines. 

### `figures`

This directory contains figures and csv files obtained when running the notebooks in the `results` directory.

## Installing environment 

To use the notebooks and launch the pipelines, you need to install the [NiPype](https://nipype.readthedocs.io/en/latest/users/install.html) Python package but also the original software package used in the pipeline (SPM, FSL, AFNI...). 

To facilitate this step, we created a Docker container based on [Neurodocker](https://github.com/ReproNim/neurodocker) that contains the necessary Python packages and software packages. To install the Docker image, two options are available.

### Option 1: Using Dockerhub
```bash
docker pull elodiegermani/open_pipeline:latest
```

The image should install itself. Once it's done you can check available images on your system:

```bash
docker images
```

### Option 2: Using a Dockerfile 
The Dockerfile used for the image stored on Dockerhub is available on the GitHub repository. But you might want to personalize your Dockerfile to install only the necessary software packages. To do so, modify the command below to modify the Dockerfile: 

```bash
docker run --rm repronim/neurodocker:0.7.0 generate docker \
           --base neurodebian:stretch-non-free --pkg-manager apt \
           --install git \
           --fsl version=6.0.3 \
           --afni version=latest method=binaries install_r=true install_r_pkgs=true install_python2=true install_python3=true \
           --spm12 version=r7771 method=binaries \
           --user=neuro \
           --workdir /home \
           --miniconda create_env=neuro \
                       conda_install="python=3.8 traits jupyter nilearn graphviz nipype scikit-image" \
                       pip_install="matplotlib" \
                       activate=True \
           --env LD_LIBRARY_PATH="/opt/miniconda-latest/envs/neuro:$LD_LIBRARY_PATH" \
           --run-bash "source activate neuro" \
           --user=root \
           --run 'chmod 777 -Rf /home' \
           --run 'chown -R neuro /home' \
           --user=neuro \
           --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' > Dockerfile
```

When you are satisfied with your Dockerfile, just build the image:

```bash
docker build --tag [name_of_the_image] - < Dockerfile
```

When the installation is finished, you have to build a container using the command below:

```bash
docker run 	-ti \
		-p 8888:8888 \
		elodiegermani/open_pipeline
```

On this command line, you need to add volumes to be able to link with your local files (original dataset and git repository). If you stored the original dataset in `data/original`, just make a volume with the `hcp_pipelines` directory:

```bash
docker run 	-ti \
		-p 8888:8888 \
		-v /users/egermani/Documents/hcp_pipelines:/home/ \
		elodiegermani/open_pipeline
``` 

After that, your container will be launched! 

### Other command that could be useful: 
#### START THE CONTAINER 

```bash
docker start [name_of_the_container]
```

#### VERIFY THE CONTAINER IS IN THE LIST 

```bash
docker ps
```

#### EXECUTE BASH OR ATTACH YOUR CONTAINER 

```bash
docker exec -ti [name_of_the_container] bash
```

**OR**

```bash
docker attach [name_of_the_container]
```

### Useful command inside the container: 
#### ACTIVATE CONDA ENVIRONMENT

```bash
source activate neuro
```

#### LAUNCH JUPYTER NOTEBOOK

```bash
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0
```

### If you did not use your container for a while: 
#### VERIFY IT STILL RUN : 

```bash
docker ps -l
```

#### IF YOUR DOCKER CONTAINER IS IN THE LIST, RUN :

```bash
docker start [name_of_the_container]
```

#### ELSE, RERUN IT WITH : 

```bash
docker run 	-ti \
		-p 8888:8888 \
		-v /home/egermani:/home \
		[name_of_the_image]
```

### To use SPM inside the container, use this command at the beginning of your script:

```python
from nipype.interfaces import spm
matlab_cmd = '/opt/spm12-r7771/run_spm12.sh /opt/matlabmcr-2010a/v713/ script'
spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)
```

## Download necessary data

Raw fMRI data used in this project are obtained from the Human Connectome Project. All imaging data and most of the behavioral data are Open Access Data, which means that these are available to those who register an account at [ConnectomeDB](db.humanconnectome.org) and agree to the [Open Access Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms). This includes agreement to comply with institutional rules and regulations. 

After logging in, chose the S1200-Release, click on "Download Images" and select "Subjects with 3T MR Session Data". On the left, chose "Unprocessed" for the Processing Level and keep only "Structural" and "Task" modalities. Select the chosen packages and download.

The original data must be stored in the `/data/original` repository.


## Reproducing subject-level analyses

Subject-level preprocessing and statistical analyses can be launched using the `src/run_pipeline.py` script. 
How to use: 
```bash
python3 run_pipeline.py -e /srv/tempdd/egermani/hcp_pipelines/data/original -r /srv/tempdd/egermani/hcp_pipelines/data/derived -s '["100206"]' -o '["l1"]' -S 'SPM' -t '["MOTOR"]' -c '["rh"]' -f 8 -p 0 -h 'derivatives'
```
This will perform the l1 analysis of subject 100206 with SPM, for contrast 'rh' (right hand) of MOTOR task. Parameters of the pipelines are: fwhm 8mm, no motion regressors and use of HRF derivatives. 

## Reproducing group-level analyses

Within-group analysis can be used to obtain group statistic maps of different pipelines and to compare these maps at a higher level. 
These can be done using the `src/run_group_analysis.py` script.
How to use:
```bash
python3 run_group_analysis.py -e /srv/tempdd/egermani/hcp_pipelines/data/derived/subject_level/"$dataset_name"/original -r /srv/tempdd/egermani/hcp_pipelines/data/derived/group_analysis/"$dataset_name" -s '["100206","100307","100410",...]' -c '["rh"]' -n 1000 -i 3
```
This will perform the within-group analysis for 1000 groups formed from the list of subjects given with i=3 subjects in each group. 

## Reproducing technical validation

Technical validation is performed to verify that statistic maps produced by the different pipelines are representative of the task of interest. 

To do it, you first need to reorganize the data and rename it using `results/run_renaming.py`. 
How to use: modify the paths to the input and output dirs and lauch:
```bash
python3 run_renaming.py
```

Technical validation can be done using the `results/run_technical_validation.py` script. 
How to use: modify the parameters in the script and launch:
```bash
python3 run_technical_validation.py
```
This will output csv files that can be used to reproduce figures. 

## Reproducing figures

Figures of technical validation can be reproduce using the Notebook `results/technical_validation.ipynb`. Do not forget to modify the path to the data. 