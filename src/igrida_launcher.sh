#!/bin/bash

output_file=$PATHLOG/$OAR_JOB_ID.txt

# Parameters
expe_name="hcp_pipelines"
main_script=/srv/tempdd/egermani/hcp_pipelines/src/run_pipeline.py

echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="/srv/tempdd/egermani/Logs/${CURRENTDATE}_OARID_${OAR_JOB_ID}/"
echo "path log :"
echo $PATHLOG
mkdir $PATHLOG

output_file=$PATHLOG/$OAR_JOB_ID.txt


e=/srv/tempdd/egermani/hcp_pipelines/data/original
r=/srv/tempdd/egermani/hcp_pipelines/data/derived
s='["100206","100408","101006"]'
o='["preprocessing"]'
S='spm'
t='["MOTOR"]'
c='["lf","rf","rh","lh","t","cue"]'
f=8

. /etc/profile.d/modules.sh

set -x

module load spack/singularity
singularity exec /srv/tempdd/egermani/open_pipeline_latest.sif /opt/miniconda-latest/bin/activate neuro
singularity exec /srv/tempdd/egermani/open_pipeline_latest.sif python3 -u $main_script -e $e -r $r -s $s -o $o -S $S -t $t -c $c -f $f >> $output_file 