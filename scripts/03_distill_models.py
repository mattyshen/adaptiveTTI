from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [0, 1, 2],
    'save_dir': [join(repo_dir, 'results')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys
params_coupled_dict = {( 
  'dataset_name', 
  'subsample_frac',
  'model_name',
  'distiller_name',
  'featurizer_name',
  'featurizer_frac',
  'featurizer_overlap',
  'depth',
  'pre_interaction',
  'pre_max_features',
  'post_interaction',
  'post_max_features'
 ):
 [(dn,
   sf,
   mn,
   din,
   fn,
   ff,
   fo,
   d,
   prei,
   premf,
   posti,
   postmf
  )
 for dn in ['ca_housing']
 for sf in [0.2]
 for mn in ["resnet", "ft_transformer"]
 for din in ["figs"]
 for fn in ["no_featurizer"]
 for ff in [0.33, 0.5, 0.75]
 for fo in [0, 1]
 for d in [2, 3]
 for prei in ["l0l2", "l1l2"]
 for premf in [0.5]
 for posti in ["l0l2"]
 for postmf in [30]
 ]}

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '02_distill_model.py'),
    actually_run=True,
    n_cpus=len(os.sched_getaffinity(0)),
)
