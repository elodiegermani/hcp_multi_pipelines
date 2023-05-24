from lib import technical_validation
from glob import glob
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=RuntimeWarning)

if __name__ == "__main__":
	soft = 'FSL'
	f = 5
	for p in [0,6,24]:
		for h in [0,1]:

			dataset = f'/srv/tempdd/egermani/pipeline_transition/data/original/stat_maps/group_level/group_50/DATASET_SOFT_{soft}_FWHM_{f}_MC_PARAM_{p}_HRF_{h}/original/*'

			maps = sorted(glob(dataset))
			print('Number of maps:', len(maps))

			df = technical_validation.run_technical_validation(maps)

			df.to_csv(f'/srv/tempdd/egermani/hcp_pipelines/figures/validation_{soft}_{f}_{p}_{h}.csv')


