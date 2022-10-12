#!/bin/bash

output_file=$PATHLOG/$OAR_JOB_ID.txt

# Parameters
expe_name="hcp_pipelines"
main_script=/srv/tempdd/egermani/hcp_pipelines/src/run_pipeline.py

echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="/srv/tempdd/egermani/Logs/${CURRENTDATE}_OARID_${OAR_JOB_ID}"

output_file=$PATHLOG/$OAR_JOB_ID.txt

e=/srv/tempdd/egermani/hcp_pipelines/data/original
r=/srv/tempdd/egermani/hcp_pipelines/data/derived
s='["101309","101915","102109","102513","102715","103010","103212","103515","104012","104820","105115","105620","106016","106521","107018","107321","107725","108121","108323","108828","109325","100307","100610","101107","101410","102008","102311","102614","102816","103111","103414","103818","104416","105014","105216","105923","106319","106824","107220","107422","108020","108222","108525","109123","109830"]'
o='["preprocessing"]'
S='spm'
t='["MOTOR"]'
c='["lf","rf","rh","lh","t","cue"]'
f=8

source /opt/miniconda-latest/etc/profile.d/conda.sh
source /opt/miniconda-latest/bin/activate
conda activate neuro
python3 $main_script -e $e -r $r -s $s -o $o -S $S -t $t -c $c -f $f