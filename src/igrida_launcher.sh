#!/bin/bash

# Parameters
expe_name="hcp_pipelines"
main_script=/srv/tempdd/egermani/hcp_pipelines/src/singularity_launcher.sh

. /etc/profile.d/modules.sh

set -x

module load spack/singularity
singularity exec /srv/tempdd/egermani/open_pipeline_latest.sif $main_script