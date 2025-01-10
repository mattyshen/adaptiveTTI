from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import numpy as np
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'save_dir': [join(repo_dir, 'results/prepro')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys

params_coupled_dict = {}
params_coupled_dict.update({('model_name',
                             'X_type',
                             'Y_type',
                             'thresh',
                             'concepts_to_edit',
                            'pre_max_features',
                            'post_max_features',
                            'max_depth',
                            'max_trees',
                            'max_rules',
                            'num_clusters'):
                            [('FIGSRegressor', xt, 'logits', t, 'random_clusters', 25, 10, 3, 30, 90, 7)
                           for xt in ['binary']
                           for t in [0, 0.5]] +
                           [('FIGSRegressor', xt, 'logits', 0, 'random_clusters', 25, 10, 3, 30, 90, 7)
                           for xt in ['global', 'cluster']]})

                           # [(m, xt, 'logits', 0, '50,89,22,52,78,82,83,100,86,20', 25, mr, 3, mt, mr, 10)
                           # for xt in ['global', 'cluster']
                           # for m in ['FIGSHydraRegressor', 'FTDHydraRegressor']
                           # for mt in [3, 5, 7]
                           # for mr in [5, 10, 15]]+
                           # [(m, xt, 'logits', t, '50,89,22,52,78,82,83,100,86,20', 25, mr, 3, mt, mr, 10)
                           # for xt in ['binary']
                           # for t in [0, 0.25, 0.5]
                           # for m in ['FIGSHydraRegressor', 'FTDHydraRegressor']
                           # for mt in [3, 5, 7]
                           # for mr in [5, 10, 15]]
                           # })

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', 'cub_distillation_edit.py'),
    actually_run=True,
    n_cpus=len(os.sched_getaffinity(0)),
)
