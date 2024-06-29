from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [0, 1, 2, 3, 4],
    'save_dir': [join(repo_dir, 'results/04_train_best_model')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys
params_coupled_dict = {}
#RF, RF+ params

# params_coupled_dict.update({('dataset_name', 
#                              'model_name', 
#                              'max_depth', 
#                              'max_features'):
#                             [(dn, mn, md, mf) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"]
#                              for mn in ['random_forest']
#                              for md in [4, 5, 6]
#                              for mf in [0.5, 0.75, 1]
#                             ] + 
#                            [(dn, mn, md, mf) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"]
#                              for mn in ['rf_plus']
#                              for md in [4, 5]
#                              for mf in [1]]})
#FIGS params
# params_coupled_dict.update({('dataset_name', 
#                              'model_name', 
#                              'max_rules', 
#                              'max_trees',
#                              'max_features'):
#                             [(dn, mn, mr, mt, mf) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"]
#                              for mn in ['figs']
#                              for mr in [60]
#                              for mt in [20, 30]
#                              for mf in [0.5, 0.75, 1]
#                             ]})
# #XGB params
# params_coupled_dict.update({('dataset_name', 
#                              'model_name', 
#                              'max_depth'):
#                             [(dn, mn, md) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"]
#                              for mn in ['xgboost']
#                              for md in [4, 5, 6]
#                             ]})
#FT, ResNet params
params_coupled_dict.update({('dataset_name', 
                             'model_name', 
                             'n_epochs'):
                            [(dn, mn, ne) 
                             for dn in ["miami_housing", "qsar", "allstate", "mercedes", "transaction"]
                             for mn in ['ft_transformer', 'resnet']
                             for ne in [100, 200]
                            ]})


# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '01_train_model.py'),
    actually_run=True,
    n_cpus=len(os.sched_getaffinity(0)),
)
