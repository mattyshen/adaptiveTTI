from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
from imodels import FIGSRegressor, FIGSClassifier

from interpretDistill.fourierDistill import FTDistillRegressorCV, FTDistillClassifierCV
from interpretDistill.featurizer import RegFeaturizer, ClassFeaturizer
from interpretDistill.tabdl import TabDLM

def get_model(task_type, model_name, args):
    if task_type == 'regression':
        if model_name == 'featurizer':
            model = RegFeaturizer(depth=args.depth, bit=args.bit)
        elif model_name == 'random_forest':
            model = RandomForestRegressor(max_depth=args.max_depth)
        elif model_name == 'figs':
            model = FIGSRegressor(max_rules=args.max_rules)
        elif model_name == 'xgboost':
            model = xgb.XGBRegressor(max_depth=args.max_depth)
        elif model_name == 'resnet':
            model = TabDLM(model_type='ResNet', 
                 task_type=task_type,
                 gpu=args.gpu,
                 n_epochs=args.n_epochs)
        elif model_name == 'ft_transformer':
            model = TabDLM(model_type='FTTransformer', 
                 task_type=task_type,
                 gpu=args.gpu,
                 n_epochs=args.n_epochs)
        elif model_name == 'ft_distill': 
            model = FTDistillRegressorCV(pre_interaction=args.pre_interaction, pre_max_features=args.pre_max_features,
                 post_interaction=args.post_interaction, post_max_features=args.post_max_features, size_interactions=args.size_interactions)
        else:
            print('Invalid model_name: {}; returning None model'.format(model_name))
            model = None
        return model
    elif task_type in ['binary', 'multiclass', 'classification']:
        if model_name == 'featurizer':
            model = ClassFeaturizer(depth=args.depth, bit=args.bit)
        elif model_name == 'random_forest':
            model = RandomForestClassifier(max_depth=args.max_depth)
        elif model_name == 'figs':
            model = FIGSClassifier(max_rules=args.max_rules)
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(max_depth=args.max_depth)
        elif model_name == 'resnet':
            model = TabDLM(model_type='ResNet', 
                 task_type=task_type,
                 gpu=args.gpu,
                 n_classes=args.n_classes,
                 n_epochs=args.n_epochs)
        elif model_name == 'ft_transformer':
            model = TabDLM(model_type='FTTransformer', 
                 task_type=task_type,
                 gpu=args.gpu,
                 n_classes=args.n_classes,
                 n_epochs=args.n_epochs)
        elif model_name == 'ft_distill': 
            model = FTDistillClassifierCV(pre_interaction=args.pre_interaction, pre_max_features=args.pre_max_features,
                 post_interaction=args.post_interaction, post_max_features=args.post_max_features, size_interactions=args.size_interactions)
        else:
            print('Invalid model_name: {}; returning None model'.format(model_name))
            model = None
        return model
    else:
        raise ValueError('Invalid task_type: {}'.format(task_type))
        return None