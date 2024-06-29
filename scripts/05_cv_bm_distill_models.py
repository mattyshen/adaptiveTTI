from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
#TODO: over many seeds?
params_shared_dict = {
    'seed': [0],
    'save_dir': [join(repo_dir, 'results/05_cv_bm_distill_model')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys

#TODO: build dictionary that maps from model to another dictionary. this dictionary maps from dataset to another dictionary. this dictionary maps from model hyper param to best model hyper param

params_coupled_dict = {}

#RF + FT_DISTILL PARAMS AND RF_PLUS + FT_DISTILL PARAMS
params_coupled_dict.update({('dataset_name', 
                             'model_name',
                             'max_depth', 
                             'max_features', 
                             'distiller_name',
                            'pre_interaction',
                            'pre_max_features',
                            'post_interaction',
                            'post_max_features'):
                            [(dn, 'random_forest', BEST_RF_MAX_DEPTH, BEST_RF_MAX_FEATURES, 'ft_distill', prei, premf, 'l0l2', 30) 
                             for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"]
                             for prei in ['l1l2', 'l0l2']
                             for premf in [0.5, 15]
                            ] +
                            [(dn, 'rf_plus', BEST_RF_PLUS_MAX_DEPTH, BEST_RF_PLUS_MAX_FEATURES, 'ft_distill', prei, premf, 'l0l2', 30) 
                             for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"]
                             for prei in ['l1l2', 'l0l2']
                             for premf in [0.5, 15]
                            ]
                           })
#RF_PLUS + FT_DISTILL PARAMS
# params_coupled_dict.update({('dataset_name', 
#                              'model_name',
#                              'max_depth', 
#                              'max_features', 
#                              'distiller_name',
#                              'pre_interaction',
#                             'pre_max_features',
#                             'post_interaction',
#                             'post_max_features'):
#                             [(dn, 'rf_plus', BEST_RF_PLUS_MAX_DEPTH, BEST_RF_PLUS_MAX_FEATURES, 'ft_distill', prei, premf, 'l0l2', 30) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"]
#                              for prei in ['l1l2', 'l0l2']
#                              for premf in [0.5, 15]
#                             ]
#                            })

#RF + FIGS PARAMS AND RF_PLUS + FIGS PARAMS
params_coupled_dict.update({('dataset_name', 
                             'model_name',
                             'max_depth', 
                             'max_features', 
                             'distiller_name',
                            'max_rules',
                            'max_trees'):
                            [(dn, 'rf_plus', BEST_RF_PLUS_MAX_DEPTH, BEST_RF_PLUS_MAX_FEATURES, 'figs', mr, mt) 
                             for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"]
                             for mr in [50, 60]
                             for mt in [20, 30]
                            ] + 
                            [(dn, 'rf_plus', BEST_RF_PLUS_MAX_DEPTH, BEST_RF_PLUS_MAX_FEATURES, 'figs', mr, mt) 
                             for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"]
                             for mr in [50, 60]
                             for mt in [20, 30]
                            ]
                           })

#RF_PLUS + FIGS PARAMS
# params_coupled_dict.update({('dataset_name', 
#                              'model_name',
#                              'max_depth', 
#                              'max_features', 
#                              'distiller_name',
#                             'max_rules',
#                             'max_trees'):
#                             [(dn, 'rf_plus', BEST_RF_PLUS_MAX_DEPTH, BEST_RF_PLUS_MAX_FEATURES, 'figs', mr, mt) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"]
#                              for mr in [50, 60]
#                              for mt in [20, 30]
#                             ]
#                            })

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '05_cv_bm_distill_model.py'),
    actually_run=True,
    n_cpus=len(os.sched_getaffinity(0)),
)
