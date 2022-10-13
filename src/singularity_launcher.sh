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
s='["120010","120111","120212","120414","120515","120717","121315","121416","121618","121719","121820","121921","122317","122418","122620","122822","123117","123420","123723","123824","123925","124220","124422","124624","124826","125222","125424","125525","126325","126426","126628","127226","127327","127630","127731","127832","127933","128026","128127","128329","128935","129028","129129","129331","129634","129937"]'
o='["preprocessing"]'
S='spm'
t='["MOTOR"]'
c='["lf","rf","rh","lh","t","cue"]'
f=8

source /opt/miniconda-latest/etc/profile.d/conda.sh
source /opt/miniconda-latest/bin/activate
conda activate neuro
python3 $main_script -e $e -r $r -s $s -o $o -S $S -t $t -c $c -f $f