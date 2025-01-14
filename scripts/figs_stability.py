from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import numpy as np
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'save_dir': [join(repo_dir, 'results/figs_stability')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys

params_coupled_dict = {}
# params_coupled_dict.update({('model_name',
#                              'X_type',
#                              'Y_type',
#                              'thresh',
#                              'concepts_to_edit',
#                             'pre_max_features',
#                             'post_max_features',
#                             'max_depth',
#                             'max_trees',
#                             'max_rules',
#                             'num_clusters'):
#                             [('FIGSRegressor', xt, 'logits', t, 'random_clusters', 25, 10, 3, 30, 90, 7)
#                            for xt in ['binary']
#                            for t in [0, 0.5]] +
#                            [('FIGSRegressor', xt, 'logits', 0, 'random_clusters', 25, 10, 3, 30, 90, 7)
#                            for xt in ['global', 'cluster']] +
#                            [('FIGSRegressor', xt, 'logits', 0, 'random_clusters', 25, 10, 3, 30, 90, 7)
#                            for xt in ['gpt1', 'gpt2', 'gpt3', 'gpt4']]})

params_coupled_dict.update({('model_name',
                             'X_type',
                             'Y_type',
                             'thresh',
                            'max_depth',
                            'max_trees',
                            'max_rules',
                            'num_clusters',
                            'model_seed',
                            'num_bootstraps'):
                            [('FIGSRegressor', 'binary', 'logits', 0.5, 3, 30, 90, 5, s, 20)
                            for s in [1, 2, 3]]})


args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', 'figs_stability.py'),
    actually_run=True,
    n_cpus=len(os.sched_getaffinity(0)),
)
