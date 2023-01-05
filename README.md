# PIPELINES COMPATIBILITY

This repository contains pipelines used to analyse HCP fMRI data with FSL and SPM with several parameters. It also contains script to perform analyses on the results of these pipelines and to study analytical variability.

## Table of contents
   * [How to cite?](#how-to-cite)
   * [Contents overview](#contents-overview)
   * [Installing environment](#installing-environment)
   * [Reproducing full analysis](#reproducing-full-analysis)
   * [Reproducing figures and tables](#reproducing-figures-and-tables)

## How to cite?


## Contents overview

### `src`

This directory contains scripts and notebooks used to launch the analysis of raw data with pipelines. 

### `data`

This directory is made to contain data that will be used by scripts/notebooks stored in the `src` directory and to contain results of those scripts. For details, check [here](#download-necessary-data).

### `results`

This directory contains notebooks and scripts that were used to analyze the results of the experiments. These notebooks were used to evaluate data compatibility between data obtained from different pipelines. 

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

## Reproducing full analysis


## Reproducing figures and tables

