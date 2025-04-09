from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

from idistill.figs_bd import FIGSBDRegressorCV, FIGSBDRegressor

def get_model(task_type, model_name, args):
    if task_type in ['regression', 'classification']:
        if model_name == 'FIGSRegressorCV':
            model = FIGSBDRegressorCV(n_rules_list=args.n_rules_list, 
                                    n_trees_list=args.n_trees_list, 
                                    n_depth_list=args.n_depth_list,
                                    min_impurity_decrease_list=args.min_impurity_decrease_list,
                                    scoring=args.metric if args.metric == 'neg_mean_squared_error' else 'r2')
        elif model_name == 'FIGSRegressor':
            model = FIGSBDRegressor(max_rules=args.max_rules, 
                                    max_trees=args.max_trees, 
                                    max_depth=args.max_depth)
        elif model_name == 'XGBRegressor':
            model = xgb.XGBRegressor(n_estimators=args.max_trees, max_depth=args.max_depth)
            
        elif model_name == 'RandomForestRegressor':
            model = RandomForestRegressor(n_estimators=args.max_trees, max_depth=args.max_depth)
        elif model_name == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(max_depth=args.max_depth)
        elif model_name == 'XGBoostRegressor':
            model = xgb.XGBRegressor(n_estimators=args.max_trees, max_depth=args.max_depth)
        else:
            model = None
        return model
    elif task_type in ['binary', 'multiclass']:
        model = None
        return model
    else:
        raise ValueError('Invalid task_type: {}'.format(task_type))
        return None