from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import numpy as np
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'save_dir': [join(repo_dir, 'results/figs_exps')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys

params_coupled_dict = {}

params_coupled_dict.update({('model_name',
                             'X_type',
                             'Y_type',
                             'max_rules', 
                             'max_trees',
                             'max_depth',
                             'task_type', 
                             'thresh'):
                           [('FIGSRegressor', xt, yt, mr, mt, md,'regression', t)
                            for xt in ['binary']
                            for yt in ['logits']
                            for mr in [15, 30, 45, 60, 75, 90]
                            for mt in [5, 10, 15, 20, 25, 30]
                            for md in [3, 4, 5]
                            for t in [0, 0.25]]+
                           [('FIGSRegressor', xt, yt, mr, mt, md,'regression', 0)
                            for xt in ['probs']
                            for yt in ['logits']
                            for mr in [15, 30, 45, 60, 75, 90]
                            for mt in [5, 10, 15, 20, 25, 30]
                            for md in [3, 4, 5]]})




args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', 'cbm_cub_distillation.py'),
    actually_run=True,
    n_cpus=len(os.sched_getaffinity(0)),
)
