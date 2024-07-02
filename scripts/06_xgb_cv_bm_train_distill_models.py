from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [0],
    'save_dir': [join(repo_dir, 'results/06_cv_bm_train_distill_model/xgb')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys

#datasets not completed = ["insurance", "qsar", "allstate", "mercedes", "transaction"]

#TODO: python /home/mattyshen/interpretableDistillation/experiments/06_cv_bm_train_distill_model.py --dataset_name concrete --model_name rf_plus --max_depth 5 --max_features 1 --distiller_name ft_distill --seed 0 --save_dir /home/mattyshen/interpretableDistillation/results/06_cv_bm_train_distill_model --use_cache 1 
params_coupled_dict = {}
#RF, RF+ params
# params_coupled_dict.update({('dataset_name', 
#                              'model_name', 
#                              'max_depth', 
#                              'max_features',
#                              'distiller_name',
#                             ):
#                            [(dn, mn, md, mf, distn) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing"]
#                              for mn in ['rf_plus']
#                              for md in [4, 5]
#                              for mf in [0.75, 1]
#                              for distn in ['ft_distill', 'figs']
#                            ]})

# params_coupled_dict.update({('dataset_name', 
#                              'model_name', 
#                              'max_depth', 
#                              'max_features',
#                              'distiller_name',
#                             ):
#                             [(dn, mn, md, mf, distn) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing"]
#                              for mn in ['random_forest', 'rf_plus']
#                              for md in [4, 5]
#                              for mf in [0.75, 1]
#                              for distn in ['ft_distill', 'figs']
#                             ]})

# params_coupled_dict.update({('dataset_name', 
#                              'model_name', 
#                              'max_depth', 
#                              'max_features',
#                              'distiller_name',
#                              'binary_mapper_name',
#                              'binary_mapper_depth',
#                              'binary_mapper_bit'
#                             ):
#                             [(dn, mn, md, mf, distn, bm, bmd, bmb) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing"]
#                              for mn in ['random_forest', 'rf_plus']
#                              for md in [4, 5]
#                              for mf in [0.75, 1]
#                              for distn in ['ft_distill', 'figs']
#                              for bm in ['dt_binary_mapper']
#                              for bmd in [2, 3]
#                              for bmb in [0]
#                             ]})
        
# #FIGS params
# params_coupled_dict.update({('dataset_name', 
#                              'model_name', 
#                              'max_rules', 
#                              'max_trees',
#                              'distiller_name'):
#                             [(dn, mn, mr, mt, distn) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing"]
#                              for mn in ['figs']
#                              for mr in [50, 60]
#                              for mt in [20, 30]
#                              for distn in ['ft_distill', 'figs']
#                             ]})
# params_coupled_dict.update({('dataset_name', 
#                              'model_name', 
#                              'max_rules', 
#                              'max_trees',
#                              'distiller_name',
#                              'binary_mapper_name',
#                              'binary_mapper_depth',
#                              'binary_mapper_bit'):
#                             [(dn, mn, mr, mt, distn, bm, bmd, bmb) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing"]
#                              for mn in ['figs']
#                              for mr in [50, 60]
#                              for mt in [20, 30]
#                              for distn in ['ft_distill', 'figs']
#                              for bm in ['dt_binary_mapper']
#                              for bmd in [2, 3]
#                              for bmb in [0]
#                             ]})
# #XGB TODO: 1 run
params_coupled_dict.update({('dataset_name', 
                             'model_name', 
                             'max_depth',
                             'distiller_name'):
                            [(dn, mn, md, distn) 
                             for dn in ["cpu_act"]
                             for mn in ['xgboost']
                             for md in [5]
                             for distn in ['ft_distill']
                            ]})
# params_coupled_dict.update({('dataset_name', 
#                              'model_name', 
#                              'max_depth',
#                              'distiller_name'):
#                             [(dn, mn, md, distn) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing"]
#                              for mn in ['xgboost']
#                              for md in [4, 5]
#                              for distn in ['ft_distill', 'figs']
#                             ]})
# params_coupled_dict.update({('dataset_name', 
#                              'model_name', 
#                              'max_depth',
#                              'distiller_name',
#                              'binary_mapper_name',
#                              'binary_mapper_depth',
#                              'binary_mapper_bit'):
#                             [(dn, mn, md, distn, bm, bmd, bmb) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing"]
#                              for mn in ['xgboost']
#                              for md in [4, 5]
#                              for distn in ['ft_distill', 'figs']
#                              for bm in ['dt_binary_mapper']
#                              for bmd in [2, 3]
#                              for bmb in [0]
#                             ]})
# #FT, ResNet params
# params_coupled_dict.update({('dataset_name', 
#                              'model_name',
#                              'distiller_name'):
#                             [(dn, mn, distn) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing"]
#                              for mn in ['ft_transformer', 'resnet']
#                              for distn in ['ft_distill', 'figs']
#                             ]})
# params_coupled_dict.update({('dataset_name', 
#                              'model_name',
#                              'distiller_name',
#                              'binary_mapper_name',
#                              'binary_mapper_depth',
#                              'binary_mapper_bit'):
#                             [(dn, mn, distn, bm, bmd, bmb) 
#                              for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing"]
#                              for mn in ['ft_transformer', 'resnet']
#                              for distn in ['ft_distill', 'figs']
#                              for bm in ['dt_binary_mapper']
#                              for bmd in [2, 3]
#                              for bmb in [0]
#                             ]})


# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '06_cv_bm_train_distill_model.py'),
    actually_run=True,
    n_cpus=len(os.sched_getaffinity(0)),
)
