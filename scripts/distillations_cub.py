from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import numpy as np
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'save_dir': [join(repo_dir, 'results/distillation_cub/cub')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys

params_coupled_dict = {}

params_coupled_dict.update({('teacher_path',
                             'train_path',
                             'test_path',
                             'student_name',
                             'max_trees',
                             'max_rules',
                             'max_depth',
                             'gpu'):
                             [('cub/outputs/best_Joint0.01_Transformer1_model_1.pth', '/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/train.pkl', '/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/test.pkl', 'DecisionTreeRegressor', 30, 200, 2)]})

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
