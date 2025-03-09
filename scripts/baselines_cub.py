from imodelsx import submit_utils
from os.path import dirname, join
import os.path
import numpy as np
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'save_dir': [join(repo_dir, 'results/distillation_cub/cub')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys

params_coupled_dict = {}

params_coupled_dict.update({('teacher_path',
                             'train_path',
                             'test_path',
                             'student_name',
                             'max_trees',
                             'max_rules',
                             'max_depth',
                             'gpu'):
                            [(path, '/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/train.pkl', '/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/test.pkl', 'XGBRegressor', 30, 1, 3, 1)
                             for path in ['cub/outputs/best_Joint0.01_Linear_model_1.pth',
                                         'cub/outputs/best_Joint0.01_MLP1_model_1.pth',
                                         'cub/outputs/best_Joint0.01_MLP2_model_1.pth',
                                         'cub/outputs/best_Joint0.01_Transformer1_model_1.pth']]+
                             [(path, '/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/train.pkl', '/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/test.pkl', 'DecisionTreeRegressor', 1, 1, md, 1)
                             for path in ['cub/outputs/best_Joint0.01_Linear_model_1.pth',
                                         'cub/outputs/best_Joint0.01_MLP1_model_1.pth',
                                         'cub/outputs/best_Joint0.01_MLP2_model_1.pth',
                                         'cub/outputs/best_Joint0.01_Transformer1_model_1.pth']
                             for md in [7, 8]]+
                           [(path, '/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/train.pkl', '/home/mattyshen/ConceptBottleneck/CUB_processed/class_attr_data_10/test.pkl', 'RandomForestRegressor', 30, 1, md, 1)
                             for path in ['cub/outputs/best_Joint0.01_Linear_model_1.pth',
                                         'cub/outputs/best_Joint0.01_MLP1_model_1.pth',
                                         'cub/outputs/best_Joint0.01_MLP2_model_1.pth',
                                         'cub/outputs/best_Joint0.01_Transformer1_model_1.pth']
                             for md in [7, 8]]})

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', 'baseline_cub.py'),
    actually_run=True,
    n_cpus=len(os.sched_getaffinity(0)),
)
