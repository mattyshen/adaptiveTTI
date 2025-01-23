from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import numpy as np
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'save_dir': [join(repo_dir, 'results/distillation_cub')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys

params_coupled_dict = {}

params_coupled_dict.update({('teacher_path',
                             'train_path',
                             'test_path',
                             'task_type',
                             'distiller_name',
                             'max_rules',
                             'max_trees',
                             'max_depth',
                             'metric'):
                            [(teacher_path, train_path, test_path, task_type, distiller_name, max_rules, max_trees, max_depth, metric)
                             for teacher_path in 
                ['/home/mattyshen/iCBM/CUB/best_models/Joint0.01SigmoidModel__Seed1/outputs/best_model_1.pth']
                             for train_path in ['/home/mattyshen/iCBM/CUB/CUB_processed/class_attr_data_10/train.pkl']
                             for test_path in ['/home/mattyshen/iCBM/CUB/CUB_processed/class_attr_data_10/test.pkl']
                             for task_type in ['regression']
                             for distiller_name in ['FIGSRegressor']
                             for max_rules in [100]
                             for max_trees in [20]
                             for max_depth in [4]
                             for metric in ['accuracy']
                            ]})


args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', 'distillation_cub.py'),
    actually_run=True,
    n_cpus=len(os.sched_getaffinity(0)),
)
