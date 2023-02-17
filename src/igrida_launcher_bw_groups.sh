#!/bin/bash

# Parameters
expe_name="pipeline_transition"
main_script=/srv/tempdd/egermani/hcp_pipelines/src/singularity_launcher_bw_groups.sh


echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="/srv/tempdd/egermani/Logs/${CURRENTDATE}_OARID_${OAR_JOB_ID}/"
echo "path log :"
echo $PATHLOG
mkdir $PATHLOG
chmod 777 $PATHLOG

. /etc/profile.d/modules.sh

set -x

module load spack/singularity
singularity exec -B /srv/tempdd/egermani /udd/egermani/Documents/open_pipeline_latest.sif $main_script