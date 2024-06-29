from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [0],
    'save_dir': [join(repo_dir, 'results/01_train_model')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys
params_coupled_dict = {( 
  'dataset_name', 
  'model_name',
  'featurizer_name',
  'featurizer_frac',
  'depth',
  'bit',
  'max_depth',
  'max_rules',
  'max_trees',
  'pre_interaction',
  'pre_max_features',
  'post_interaction',
  'post_max_features'
 ):
 [(dn,
   mn,
   fn,
   ff,
   d,
   b,
   md, 
   mr, 
   mt,
   prei,
   premf,
   posti,
   postmf
  )
 for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"]
 for mn in ["random_forest", "rf_plus", "xgboost", "resnet", "ft_transformer"]
 for fn in ["no_featurizer", "featurizer"]
 for ff in [0.3, 0.7]
 for d in [2, 3]
 for b in [0, 1]
 for md in [4, 5]
 for mr in [1]
 for mt in [1]
 for prei in ["l0l2"]
 for premf in [0]
 for posti in ["l0l2"]
 for postmf in [0]
 ] + 
 [(dn,
   mn,
   fn,
   ff,
   d,
   b,
   md, 
   mr, 
   mt,
   prei,
   premf,
   posti,
   postmf
  )
 for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"]
 for mn in ["figs"]
 for fn in ["no_featurizer", "featurizer"]
 for ff in [0.3, 0.7]
 for d in [2, 3]
 for b in [0, 1]
 for md in [1]
 for mr in [50, 60]
 for mt in [30]
 for prei in ["l0l2"]
 for premf in [0]
 for posti in ["l0l2"]
 for postmf in [0]
 ] + 
 [(dn,
   mn,
   fn,
   ff,
   d,
   b,
   md, 
   mr, 
   mt,
   prei,
   premf,
   posti,
   postmf
  )
 for dn in ["ca_housing", "abalone", "parkinsons", "airfoil", "cpu_act", "concrete", "powerplant", "miami_housing", "insurance", "qsar", "allstate", "mercedes", "transaction"]
 for mn in ["ft_distill"]
 for fn in ["featurizer"]
 for ff in [0.3, 0.7]
 for d in [2, 3]
 for b in [0, 1]
 for md in [1]
 for mr in [1]
 for mt in [1]
 for prei in ["l1l2", "l0l2"]
 for premf in [0.5]
 for posti in ["l0l2"]
 for postmf in [25, 30]
 ]
}

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
