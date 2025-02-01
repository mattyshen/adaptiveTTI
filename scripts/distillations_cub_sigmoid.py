from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import numpy as np
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'save_dir': [join(repo_dir, 'results/distillation_cub/cub_sigmoid')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys

params_coupled_dict = {}

# params_coupled_dict.update({('teacher_path',
#                              'train_path',
#                              'test_path',
#                              'task_type',
#                              'distiller_name',
#                              'max_rules',
#                              'max_trees',
#                              'max_depth',
#                              'metric'):
#                             [(teacher_path, train_path, test_path, task_type, distiller_name, max_rules, max_trees, max_depth, metric)
#                              for teacher_path in 
#                 [f'/home/mattyshen/iCBM/CUB/best_models/Joint0.01SigmoidModel__Seed{s}/outputs/best_model_{s}.pth' for s in range(1, 4)]
#                              for train_path in ['/home/mattyshen/iCBM/CUB/CUB_processed/class_attr_data_10/train.pkl']
#                              for test_path in ['/home/mattyshen/iCBM/CUB/CUB_processed/class_attr_data_10/test.pkl']
#                              for task_type in ['regression']
#                              for distiller_name in ['FIGSRegressor']
#                              for max_rules in [100]
#                              for max_trees in [20]
#                              for max_depth in [4]
#                              for metric in ['accuracy']
#                             ]})

#no-sigmoid model
params_coupled_dict.update({('teacher_path',
                             'train_path',
                             'test_path',
                             'gpu'):
                            [('cub_sigmoid/outputs/best_model_large_1.pth', train_path, test_path, 0)
                             for train_path in ['/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/train.pkl']
                             for test_path in ['/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/test.pkl']
                            ]+
                           [('cub_sigmoid/outputs/best_model_med_1.pth', train_path, test_path, 1)
                             for train_path in ['/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/train.pkl']
                             for test_path in ['/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/test.pkl']
                            ]+
                           [('cub_sigmoid/outputs/best_model_small_1.pth', train_path, test_path, 2)
                             for train_path in ['/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/train.pkl']
                             for test_path in ['/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/test.pkl']
                            ]})

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', 'distillation_cub_sigmoid.py'),
    actually_run=True,
    n_cpus=len(os.sched_getaffinity(0)),
)
