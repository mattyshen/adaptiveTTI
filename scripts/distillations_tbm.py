from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import numpy as np
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'save_dir': [join(repo_dir, 'results/test/cebab')],
    'use_cache': [0], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys

params_coupled_dict = {}

params_coupled_dict.update({('train_path',
                             'task_type',
                             'task_name',
                             'metric',
                             'do_intervention',
                             'teacher_name',
                             'student_name',
                             'n_rules_list',
                             'n_trees_list',
                             'n_depth_list',
                             'num_interactions_intervention'):
                             [('data/tbm_gpt4', 'regression', 'cebab', 'r2', 'True', 'linear', 'FIGSRegressorCV', '100,200,250', '20,30,50', '3,4', '0')]})
                             #[('data/tbm_gpt4', 'classification', 'agnews', 'accuracy', 'True', 'linear', 'FIGSRegressor', '3')]})

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', 'distillation_tbm.py'),
    actually_run=True,
    n_cpus=len(os.sched_getaffinity(0)),
)
