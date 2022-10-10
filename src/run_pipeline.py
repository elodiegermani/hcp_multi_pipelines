import random
import os 
from os.path import join as opj
import sys
import getopt
import json

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=RuntimeWarning)

if __name__ == "__main__":
    subject_list = []
    exp_dir = None
    result_dir = None
    software = None
    operation = None
    task_list = []
    contrast_list = []
    fwhm_list=[]

    try:
        OPTIONS, REMAINDER = getopt.getopt(sys.argv[1:], 'e:r:s:o:S:t:c:f:', ['exp_dir=', 'result_dir=', 'subjects=', 'operation=',
            'software=', 'task=', 'contrast=', 'fwhm='])

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    # Replace variables depending on options
    for opt, arg in OPTIONS:
        if opt in ('-e', '--exp_dir'):
            exp_dir= arg
        elif opt in ('-r', '--result_dir'):
            result_dir = arg
        elif opt in ('-s', '--subjects'):
            subject_list = json.loads(arg)
        elif opt in ('-o', '--operation'): 
            operation = json.loads(arg) # preprocess, l1
        elif opt in ('-S', '--software'):
            software = str(arg)
        elif opt in ('-t', '--task'):
            task_list = json.loads(arg)
        elif opt in ('-c', '--contrast'):
            contrast_list = json.loads(arg)
        elif opt in ('-f', '--fwhm'):
            fwhm_list.append(int(arg))

    print('OPTIONS   :', OPTIONS)

    if len(operation) == 0:
        print('All operations will be performed.')
        operation = ['preprocessing', 'l1']

    package = 'lib.' + software + '_pipelines'
    pipeline = importlib.import_module(package)

    working_dir = f'{software}_preprocessing/intermediate_results'
    output_dir = f'{software}_preprocessing'

    if 'preprocessing' in operation:
        preprocess = pipeline.get_preprocessing(exp_dir, result_dir, working_dir, output_dir, subject_list, task_list, 
                                                fwhm_list)
        preprocess.run('MultiProc', plugin_args={'n_procs': 8})

    if 'l1' in operation:
        l1_analysis = pipeline.get_l1_analysis(exp_dir, output_dir, working_dir, result_dir, subject_list, task_list, 
                                                contrast_list, fwhm_list)
        l1_analysis.run('MultiProc', plugin_args={'n_procs': 8})

        if software == 'fsl':
            registration = pipeline.get_registration(exp_dir, output_dir, working_dir, result_dir, subject_list, task_list, 
                contrast_list, fwhm_list)

            registration.run('MultiProc', plugin_args={'n_procs': 8})






