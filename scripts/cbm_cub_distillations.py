from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import numpy as np
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'save_dir': [join(repo_dir, 'results/cbm_cub')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys

params_coupled_dict = {}
# params_coupled_dict.update({('model_name',
#                              'X_type',
#                              'Y_type',
#                              'max_rules', 
#                              'max_trees',
#                              'max_depth',
#                              'task_type'):
#                             [('FIGSRegressor', xt, yt, mr, mt, md,'regression')
#                             for xt in ['probs', 'binary']
#                             for yt in ['probs', 'logits']
#                             for mr in [60, 90]
#                             for mt in [15, 30]
#                             for md in [3, 4]]+
#                             [('FIGSHydraRegressor', xt, yt, mr, mt, md,'regression')
#                             for xt in ['probs', 'binary']
#                             for yt in ['probs', 'logits']
#                             for mr in [5, 10, 15]
#                             for mt in [3, 5]
#                             for md in [3, 4]]+
#                            [('FIGSClassifier', xt, yt, mr, mt, md,'classification')
#                             for xt in ['probs', 'binary']
#                             for yt in ['classes']
#                             for mr in [60, 90]
#                             for mt in [15, 30]
#                             for md in [3, 4]]})
# params_coupled_dict.update({('model_name',
#                              'X_type',
#                              'Y_type',
#                              'max_rules', 
#                              'max_trees',
#                              'max_depth',
#                              'task_type'):
#                             [('FIGSHydraRegressor', xt, yt, mr, mt, md,'regression')
#                             for xt in ['probs', 'binary']
#                             for yt in ['probs', 'logits']
#                             for mr in [5, 10, 15]
#                             for mt in [3, 5]
#                             for md in [3, 4]]})

# params_coupled_dict.update({('model_name',
#                              'X_type',
#                              'Y_type',
#                              'max_rules', 
#                              'max_trees',
#                              'max_depth',
#                              'task_type',
#                              'thresh',
#                              'reg_depth',
#                              'shrink_depth',
#                              'reg_shrink'):
#                             [('FIGSRegressor', 'binary', 'logits', mr, mt, md,'regression', 0.5, rd, 0, 0)
#                             for mr in [60, 90]
#                             for mt in [15, 30]
#                             for md in [4, 5]
#                             for rd in [0, 1]]})

# #FTDistillClassifier
# params_coupled_dict.update({('model_name',
#                              'X_type',
#                              'Y_type',
#                              'max_depth',
#                              'pre_interaction',
#                              'pre_max_features',
#                              'post_interaction',
#                              'post_max_features',
#                              'task_type'):
#                            [('ft_distill', 'binary', 'classes', 3, 'l0l2', pm, 'l0l2', pmf, 'classification')
#                             for pm in [75, 112]
#                             for pmf in [30, 50, 75]]+
#                            [('ft_distill', 'binary', 'classes', 4, 'l0l2', 40, 'l0l2', pmf, 'classification')
#                            for pmf in [30, 50, 75]]})
#DT + RF
# params_coupled_dict.update({('model_name',
#                              'X_type',
#                              'Y_type',
#                              'max_depth',
#                              'task_type'):
#                            [(mn, xt, yt, md, 'classification')
#                             for mn in ['decision_tree', 'random_forest']
#                             for xt in ['probs', 'binary']
#                             for yt in ['classes']
#                             for md in [7, 8]]+
#                            [(mn, xt, yt, md, 'regression')
#                             for mn in ['decision_tree', 'random_forest']
#                             for xt in ['probs', 'binary']
#                             for yt in ['probs', 'logits']
#                             for md in [7, 8]]})

#ElasticNet, LogisticRegression
# params_coupled_dict.update({('model_name',
#                              'X_type',
#                              'Y_type',
#                              'task_type'):
#                            [(mn, xt, yt, 'regression')
#                             for mn in ['elastic_net']
#                             for xt in ['probs', 'binary']
#                             for yt in ['probs', 'logits']]+
#                            [(mn, xt, yt, 'classification')
#                             for mn in ['log_reg']
#                             for xt in ['probs', 'binary']
#                             for yt in ['classes']]})
params_coupled_dict.update({('model_name',
                             'X_type',
                             'Y_type',
                             'thresh',
                             'max_depth',
                             'task_type'):
                           [('xgboost', xt, yt, 0, md, tt)
                            for xt in ['probs']
                            for yt in ['probs', 'logits']
                            for md in [3, 8]
                            for tt in ['regression']]+
                           [('xgboost', xt, yt, t, md, tt)
                            for xt in ['binary']
                            for yt in ['probs', 'logits']
                            for t in [0, 0.25, 0.5]
                            for md in [3, 8]
                            for tt in ['regression']]+
                           [('xgboost', xt, yt, 0, md, tt)
                            for xt in ['probs']
                            for yt in ['classes']
                            for md in [3, 8]
                            for tt in ['classification']]+
                           [('xgboost', xt, yt, t, md, tt)
                            for xt in ['binary']
                            for yt in ['classes']
                            for t in [0, 0.25, 0.5]
                            for md in [3, 8]
                            for tt in ['classification']]})
# FTDistillRegressor MO L0L2
# params_coupled_dict.update({('model_name',
#                              'X_type',
#                              'Y_type',
#                              'max_depth',
#                              'pre_interaction',
#                              'post_interaction',
#                              'post_max_features',
#                              'task_type',
#                              'mo'):
#                            [('ft_distill', xt, yt, 3, 'None', 'l0l2', pmf, 'regression', True)
#                             for xt in ['probs']
#                             for yt in ['probs']
#                             for pmf in [5, 7]]})


# params_coupled_dict.update({('model_name',
#                              'X_type',
#                              'Y_type',
#                              'max_depth',
#                              'pre_interaction',
#                              'pre_max_features',
#                              'post_interaction',
#                              'post_max_features',
#                              'task_type',
#                              'thresh'):
#                            [('ft_distill', xt, 'classes', 1, 'l0l2', pm, 'l0l2', pmf, 'classification', t)
#                             for xt in ['binary', 'probs']
#                             for pm in [112]
#                             for pmf in [112]
#                             for t in [0, 0.25, 0.5]]+
#                             [('ft_distill', xt, 'classes', 3, 'l0l2', pm, 'l0l2', pmf, 'classification', t)
#                             for xt in ['binary', 'probs']
#                             for pm in [50, 75]
#                             for pmf in [75]
#                             for t in [0, 0.25, 0.5]]})

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
