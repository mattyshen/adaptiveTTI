from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
from imodels import FIGSRegressor, FIGSClassifier

from fourierDistill import FTDistillRegressorCV, FTDistillClassifierCV
from featurizer import RegFeaturizer, ClassFeaturizer
from tabdl import TabDLM

def get_model(args):
    if args.task_type == 'regression':
        if args.model_name == 'featurizer':
            model = RegFeaturizer(depth=args.depth, bit=args.bit)
        elif args.model_name == 'random_forest':
            model = RandomForestRegressor(max_depth=args.max_depth)
        elif args.model_name == 'figs':
            model = FIGSRegressor(max_rules=args.max_rules)
        elif args.model_name == 'xgboost':
            model = xgb.XGBRegressor(max_depth=args.max_depth)
        elif args.model_name == 'resnet':
            model = TabDLM(model_type='ResNet', 
                 task_type=args.task_type,
                 n_classes=args.n_classes,
                 n_cont_features=args.n_cont_features, 
                 cat_cardinalities=args.cat_cardinalities, 
                 d_out=args.d_out,
                 n_epochs=args.n_epochs,
                 patience=16,
                 batch_size=512)
        elif args.model_name == 'ft_transformer':
            model = TabDLM(model_type='FTTransformer', 
                 task_type=args.task_type,
                 n_classes=args.n_classes,
                 n_cont_features=args.n_cont_features, 
                 cat_cardinalities=args.cat_cardinalities, 
                 d_out=args.d_out,
                 n_epochs=args.n_epochs,
                 patience=16,
                 batch_size=512)
        elif args.model_name == 'ft_distill': 
            model = FTDistillRegressorCV(pre_interaction=args.pre_interaction, pre_max_features=args.pre_max_features,
                 post_interaction=args.post_interaction, pre_max_features=args.post_max_features)
        else:
            print('Invalid model_name: {}; returning None model'.format(args.model_name))
            model = None
        return model
    elif args.task_type == 'classification':
        if args.model_name == 'featurizer':
            model = ClassFeaturizer(depth=args.depth, bit=args.bit)
        elif args.model_name == 'random_forest':
            model = RandomForestClassifier(max_depth=args.max_depth)
        elif args.model_name == 'figs':
            model = FIGSClassifier(max_rules=args.max_rules)
        elif args.model_name == 'xgboost':
            model = xgb.XGBClassifier(max_depth=args.max_depth)
        elif args.model_name == 'resnet':
            model = TabDLM(model_type='ResNet', 
                 task_type=args.task_type,
                 n_classes=args.n_classes,
                 n_cont_features=args.n_cont_features, 
                 cat_cardinalities=args.cat_cardinalities, 
                 d_out=args.d_out,
                 n_epochs=args.n_epochs,
                 patience=16,
                 batch_size=512)
        elif args.model_name == 'ft_transformer':
            model = TabDLM(model_type='FTTransformer', 
                 task_type=args.task_type,
                 n_classes=args.n_classes,
                 n_cont_features=args.n_cont_features, 
                 cat_cardinalities=args.cat_cardinalities, 
                 d_out=args.d_out,
                 n_epochs=args.n_epochs,
                 patience=16,
                 batch_size=512)
        elif args.model_name == 'ft_distill': 
            model = FTDistillClassifierCV(pre_interaction=args.pre_interaction, pre_max_features=args.pre_max_features,
                 post_interaction=args.post_interaction, pre_max_features=args.post_max_features)
        else:
            print('Invalid model_name: {}; returning None model'.format(args.model_name))
            model = None
        return model
    else:
        raise ValueError('Invalid task_type: {}'.format(args.task_type))
        return None