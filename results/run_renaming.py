from lib import rename 

def main():
	out_dir = '/srv/tempdd/egermani/hcp_many_pipelines'

	# FSL rename
	in_dir = '/srv/tempdd/egermani/hcp_pipelines/data/derived/fsl_preprocessing/registration_fsl'
	rename.rename_fsl(in_dir, out_dir)

	# SPM rename
	in_dir = '/srv/tempdd/egermani/hcp_pipelines/data/derived/spm_preprocessing/l1_analysis_spm'
	rename.rename_spm(in_dir, out_dir)

	# Group rename
	in_dir = '/srv/tempdd/egermani/hcp_pipelines/data/derived/group_analysis'
	rename.rename_group(in_dir, out_dir)

if __name__ == '__main__':
	main()
