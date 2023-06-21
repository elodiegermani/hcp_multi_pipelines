from lib import rename 

def main():
	in_dir = '/srv/tempdd/egermani/hcp_pipelines/data/derived/group_analysis'
	out_dir = '/srv/tempdd/egermani/hcp_many_pipelines'
	rename.rename_group(in_dir, out_dir)

if __name__ == '__main__':
	main()
