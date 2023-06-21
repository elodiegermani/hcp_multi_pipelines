expe_name="rename"
main_script=/srv/tempdd/egermani/hcp_pipelines/results/run_renaming.py


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

/srv/tempdd/egermani/workEnv/bin/python -u $main_script