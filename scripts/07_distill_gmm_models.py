from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [0],
    'save_dir': [join(repo_dir, 'results/07_distill_gmm_models')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys
params_coupled_dict = {}
#RF, RF+ params

# params_coupled_dict.update({('dataset_name', 
#                              'model_name', 
#                              'max_depth', 
#                              'max_features',
#                              'distiller_name'):
#                             [(dn, mn, md, mf, d_n) 
#                              for dn in ["ca_housing", "miami_housing"]
#                              for mn in ['random_forest']
#                              for md in [5, 6]
#                              for mf in [1]
#                              for d_n in ['ft_distill', 'figs']
#                             ] + 
#                            [(dn, mn, md, mf, d_n) 
#                              for dn in ["ca_housing", "miami_housing"]
#                              for mn in ['rf_plus']
#                              for md in [4]
#                              for mf in [1]
#                              for d_n in ['ft_distill', 'figs']
#                            ]})
#FIGS params
params_coupled_dict.update({('dataset_name', 
                             'model_name', 
                             'max_rules', 
                             'max_trees',
                             'max_features',
                             'distiller_name'):
                            [(dn, mn, mr, mt, mf, d_n) 
                             for dn in ["ca_housing", "miami_housing"]
                             for mn in ['figs']
                             for mr in [60]
                             for mt in [20, 30]
                             for mf in [1]
                             for d_n in ['ft_distill', 'figs']
                            ]})
# #XGB params
params_coupled_dict.update({('dataset_name', 
                             'model_name', 
                             'max_depth',
                            'distiller_name'):
                            [(dn, mn, md, d_n) 
                             for dn in ["ca_housing", "miami_housing"]
                             for mn in ['xgboost']
                             for md in [5, 6]
                             for d_n in ['ft_distill', 'figs']
                            ]})
#FT, ResNet params
params_coupled_dict.update({('dataset_name', 
                             'model_name', 
                             'n_epochs', 
                            'distiller_name'):
                            [(dn, mn, ne, d_n) 
                             for dn in ["ca_housing", "miami_housing"]
                             for mn in ['ft_transformer', 'resnet']
                             for ne in [100]
                             for d_n in ['ft_distill', 'figs']
                            ]})


# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '07_distill_gmm_model.py'),
    actually_run=True,
    n_cpus=len(os.sched_getaffinity(0)),
)
