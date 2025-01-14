from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import numpy as np
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'save_dir': [join(repo_dir, 'results/cub_hydra')],
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
                            [(m, xt, 'logits', 0.5, 'random_clusters', 20, mt*2, 3, mt, mr, 7)
                           for xt in ['binary', 'global', 'cluster', 'gpt1', 'gpt2', 'gpt3', 'gpt4']
                           for m in ['FIGSHydraRegressor', 'FTDHydraRegressor']
                           for mt in [5, 7, 10]
                           for mr in [10, 15, 20]]})

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

# params_coupled_dict.update({('model_name',
#                              'X_type',
#                              'Y_type',
#                              'thresh',
#                              'concepts_to_edit',
#                             'pre_max_features',
#                             'post_max_features',
#                             'max_depth',
#                             'max_trees',
#                             'max_rules'):
#                            [(m, 'binary', 'logits', t, '50,89,22,52,78,82,83,100,86,20', premf, postmf, md, 30, 60)
#                            for m in ['FTDHydraRegressorCV']
#                            for t in [0, 0.5]
#                            for premf in [10, 25]
#                            for postmf in [7]
#                            for md in [3, 4]]+
#                            [(m, 'binary', 'logits', t, '50,89,22,52,78,82,83,100,86,20', 1, 30, md, mt, mr)
#                            for m in ['FIGSRegressor']
#                            for t in [0, 0.5]
#                            for md in [3, 4]
#                            for mt in [60, 75]
#                            for mr in [60, 90]]})

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
