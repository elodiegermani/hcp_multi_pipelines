#!/bin/bash

output_file=$PATHLOG/$OAR_JOB_ID.txt

# Parameters
expe_name="pipeline_transition"
main_script=/srv/tempdd/egermani/hcp_pipelines/src/group_map_comparison.py

echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="/srv/tempdd/egermani/Logs/${CURRENTDATE}_OARID_${OAR_JOB_ID}"

output_file=$PATHLOG/$OAR_JOB_ID.txt

dataset_name=DATASET_SOFT_SPM_FWHM_8_MC_PARAM_6

e=/srv/tempdd/egermani/pipeline_transition/data/original/"$dataset_name"/original


r=/srv/tempdd/egermani/hcp_pipelines/data/derived/group_analysis/"$dataset_name"
S='["101006","105115","106319","109325","113316","114419","115017","117728","119126","120515","123117","130316","134223","134627","135124","137936","138332","140824","142424","144731","146129","146533","148335","151425","151526","152225","153934","154431","158035","158136","159441","160123","169040","169444","169545","172130","173637","174437","175035","175237","178142","180230","183337","185139","187143","188448","191336","192439","195445","198047"]'
c='["rh","rf","lh","lf","cue","t"]'
i=1
n=50

source /opt/miniconda-latest/etc/profile.d/conda.sh
source /opt/miniconda-latest/bin/activate
conda activate neuro

for fwhm in 5
do
	for mc in 6 24
	do
		for hrf in 0 1
		do
			dataset_name="DATASET_SOFT_SPM_FWHM_${fwhm}_MC_PARAM_${mc}_HRF_${hrf}"
			e=/srv/tempdd/egermani/pipeline_transition/data/original/"$dataset_name"/original
			r=/srv/tempdd/egermani/hcp_pipelines/data/derived/group_analysis/"$dataset_name"
			python3 $main_script -e $e -r $r -S $S -c $c --n_iter $i --n_sub $n 
done
done
done