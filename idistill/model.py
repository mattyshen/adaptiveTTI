from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

#from imodels import FIGSRegressor, FIGSClassifier

from idistill.whitebox_figs import FIGSHydraRegressor, FIGSRegressor, FIGSClassifier
from idistill.ftd import FTDistillRegressor, FTDistillRegressorCV, FTDistillClassifierCV

from idistill.subset_predictors import L0L2RegressorCV

def get_model(task_type, model_name, args):
    if task_type == 'regression':
        if model_name == 'FIGSHydraRegressor':
            #model = MultiOutputRegressor(FIGSRegressor(max_rules=args.max_rules, max_trees=args.max_trees, max_depth=args.max_depth))
            model = FIGSHydraRegressor(max_rules=args.max_rules, max_trees=args.max_trees, max_depth=args.max_depth)
        elif model_name == 'FIGSRegressor':
            model = FIGSRegressor(max_rules=args.max_rules, max_trees=args.max_trees, max_depth=args.max_depth)
        elif model_name == 'XGBoostRegressor':
            model = xgb.XGBRegressor(n_estimators=args.max_trees, max_depth=args.max_depth)
        elif model_name == 'FTDHydraRegressor':
            model = MultiOutputRegressor(FTDistillRegressor(pre_interaction=args.pre_interaction, pre_max_features=args.pre_max_features,
                 post_interaction=args.post_interaction, post_max_features=args.post_max_features, size_interactions=args.max_depth))
        elif model_name == 'FTDHydraRegressorCV':
            model = MultiOutputRegressor(FTDistillRegressorCV(pre_interaction=args.pre_interaction, pre_max_features=args.pre_max_features,
                 post_interaction=args.post_interaction, post_max_features=args.post_max_features, size_interactions=args.max_depth))
        elif model_name == 'FTDRegressorCV': 
            model = FTDistillRegressorCV(pre_interaction=args.pre_interaction, pre_max_features=args.pre_max_features,
                 post_interaction=args.post_interaction, post_max_features=args.post_max_features, size_interactions=args.max_depth, mo=args.mo)
        else:
            model = None
        return model
    elif task_type in ['binary', 'multiclass', 'classification']:
        if model_name == 'FIGSClassifier':
            model = FIGSClassifier(max_rules=args.max_rules, max_trees=args.max_trees, max_depth=args.max_depth)
        elif model_name == 'XGBClassifier':
            model = xgb.XGBClassifier(n_estimators=args.max_trees, max_depth=args.max_depth)
        elif model_name == 'FTDClassifierCV': 
            model = FTDistillClassifierCV(pre_interaction=args.pre_interaction, pre_max_features=args.pre_max_features,
                 post_interaction=args.post_interaction, post_max_features=args.post_max_features, size_interactions=args.max_depth)
        else:
            model = None
        return model
    else:
        raise ValueError('Invalid task_type: {}'.format(task_type))
        return None