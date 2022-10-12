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
s='["110007","111009","111413","112112","112819","113316","113821","114217","114621","115017","115724","116221","116726","117324","118023","118528","118932","119732","110411","111211","111514","112314","112920","113417","113922","114318","114823","115219","115825","116423","117021","117728","118124","118730","119025","119833","110613","111312","111716","112516","113215","113619","114116","114419","114924","115320","116120","116524","117122","117930","118225","118831","119126"]'
o='["preprocessing"]'
S='spm'
t='["MOTOR"]'
c='["lf","rf","rh","lh","t","cue"]'
f=8

source /opt/miniconda-latest/etc/profile.d/conda.sh
source /opt/miniconda-latest/bin/activate
conda activate neuro
python3 $main_script -e $e -r $r -s $s -o $o -S $S -t $t -c $c -f $f