#!/bin/bash

# Parameters
expe_name="technical_validation"
main_script=/srv/tempdd/egermani/hcp_pipelines/results/run_technical_validation.py


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
module load miniconda

source /soft/igrida/miniconda/miniconda-latest/bin/activate /srv/tempdd/egermani/workEnv

#conda activate workEnv

# -u : Force les flux de sortie et d'erreur standards à ne pas utiliser de tampon. 
# Cette option n'a pas d'effet sur le flux d'entrée standard
/srv/tempdd/egermani/workEnv/bin/python -u $main_script