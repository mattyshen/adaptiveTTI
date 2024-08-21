from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import numpy as np
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [0],
    'save_dir': [join(repo_dir, 'results/08_timed')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys

#datasets not completed = ["insurance", "qsar", "allstate", "mercedes", "transaction"]

params_coupled_dict = {}
params_coupled_dict.update({('dataset_name',
                             'binary_mapper_frac',
                             'max_rules', 
                             'max_trees',
                             'max_depth',
                             'n_estimators',
                             'pre_max_features',
                             'max_min_interaction_size'):
                            [(dn, bmf, mr, mt, int(np.ceil(np.log2(mr/mt))), mt, pmf, um) 
                             for dn in ["abalone", "parkinsons", "airfoil",  "powerplant", "miami_housing", "insurance", "qsar"]
                             for bmf in [0]
                             for mr in [20, 30]
                             for mt in [5, 10]
                             for pmf in [0.5, 1]
                             for um in [3]
                            ]+
                           [(dn, bmf, mr, mt, int(np.ceil(np.log2(mr/mt))), mt, pmf, um) 
                             for dn in ["abalone", "parkinsons", "airfoil",  "powerplant", "miami_housing", "insurance", "qsar"]
                             for bmf in [0]
                             for mr in [20, 30]
                             for mt in [5, 10]
                             for pmf in [0.25, 0.5]
                             for um in [5]
                            ]})
# params_coupled_dict.update({('dataset_name',
#                              'binary_mapper_frac',
#                              'max_rules', 
#                              'max_trees',
#                              'pre_max_features',
#                              'use_min'):
#                             [(dn, bmf, mr, mt, pmf, um) 
#                              for dn in ["abalone", "parkinsons", "airfoil",  "powerplant", "miami_housing", "insurance", "qsar"]
#                              for bmf in [0]
#                              for mr in [20]
#                              for mt in [5, 10]
#                              for pmf in [0.5, 1]
#                              for um in [0, 1]
#                             ]})

# params_coupled_dict.update({('dataset_name',
#                              'binary_mapper_frac',
#                              'max_rules', 
#                              'max_trees',
#                              'pre_max_features',
#                              'use_min'):
#                             [(dn, bmf, mr, mt, pmf, um) 
#                              for dn in ["abalone", "parkinsons", "airfoil",  "powerplant", "miami_housing", "insurance", "qsar"]
#                              for bmf in [0]
#                              for mr in [10, 20]
#                              for mt in [5, 10]
#                              for pmf in [0.5, 1]
#                              for um in [0, 1]
#                             ]})

# params_coupled_dict.update({('dataset_name',
#                              'binary_mapper_frac',
#                              'max_rules', 
#                              'max_trees',
#                              'pre_max_features', 
#                              'use_min'):
#                             [(dn, 0, mr, mt, 1, um) 
#                              for dn in ["abalone", "parkinsons", "airfoil", "qsar", "allstate", "powerplant", "miami_housing", "insurance"]
#                              for mr in [15, 20]
#                              for mt in [5, 10, 15]
#                              for um in [0]
#                             ]})
# params_coupled_dict.update({('dataset_name',
#                              'binary_mapper_frac',
#                              'max_rules', 
#                              'max_trees',
#                              'pre_max_features', 
#                              'use_min'):
#                             [(dn, 0, mr, mt, 1, um) 
#                              for dn in ["mercedes", "transaction"]
#                              for mr in [15, 20]
#                              for mt in [5, 10, 15]
#                              for um in [0, 1]
#                             ]})
# params_coupled_dict.update({('dataset_name',
#                              'binary_mapper_frac',
#                              'max_rules', 
#                              'max_trees',
#                              'pre_max_features', 
#                              'use_min'):
#                             [(dn, 0, mr, mt, 1, um) 
#                              for dn in ["cpu_act", "concrete"]
#                              for mr in [15, 20]
#                              for mt in [5, 10, 15]
#                              for um in [0, 1]
#                             ]})
# params_coupled_dict.update({('dataset_name',
#                              'binary_mapper_frac',
#                              'max_rules', 
#                              'max_trees',
#                              'pre_max_features'):
#                             [(dn, bmf, mr, mt, pmf) 
#                              for dn in ["ca_housing"]
#                              for bmf in [0.5]
#                              for mr in [20]
#                              for mt in [15]
#                              for pmf in [1]
#                             ]})

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '08_figs_restructure.py'),
    actually_run=True,
    n_cpus=len(os.sched_getaffinity(0)),
)
