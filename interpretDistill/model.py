from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import MultiTaskElasticNetCV, LogisticRegressionCV
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
#from imodels import FIGSRegressor, FIGSClassifier
from imodels.importance import RandomForestPlusRegressor, RandomForestPlusClassifier

from interpretDistill.mo_figs import FIGSHydraRegressor, FIGSRegressor, FIGSClassifier
from interpretDistill.fourierDistill import FTDistillRegressorCV, FTDistillClassifierCV
from interpretDistill.binary_mapper import DTRegBinaryMapper, DTClassBinaryMapper, GMMBinaryMapper, FIGSBinaryMapper
from interpretDistill.tabdl import TabDLM
from interpretDistill.subset_predictors import L0L2RegressorCV

def get_model(task_type, model_name, args):
    if task_type == 'regression':
        if model_name == 'FIGSHydraRegressor':
            model = FIGSHydraRegressor(max_rules=args.max_rules, max_trees=args.max_trees, max_depth=args.max_depth)
        elif model_name == 'FIGSRegressor':
            model = FIGSRegressor(max_rules=args.max_rules, max_trees=args.max_trees, max_depth=args.max_depth)
        elif model_name == 'dt_binary_mapper':
            model = DTRegBinaryMapper(depth=args.binary_mapper_depth, bit=args.binary_mapper_bit)
            params = ['bit', 'depth']
        elif model_name == 'gmm_binary_mapper':
            model = GMMBinaryMapper(max_gmm_components=2**args.binary_mapper_depth)
            params = []
        elif model_name == 'figs_binary_mapper':
            model = FIGSBinaryMapper(figs=FIGSRegressor(max_rules=args.max_rules, max_trees=args.max_trees, max_features=args.max_features))
            params = []
        elif model_name == 'decision_tree':
            model = DecisionTreeRegressor(max_depth=args.max_depth, min_samples_leaf=5)
        elif model_name == 'random_forest':
            model = RandomForestRegressor(max_depth=args.max_depth, min_samples_leaf=5)
            params = ['max_depth', 'max_features']
        elif model_name == 'rf_plus':
            rf_model = RandomForestRegressor(max_depth=args.max_depth, min_samples_leaf=5, max_features=args.max_features)
            model = RandomForestPlusRegressor(rf_model=rf_model)
            params = ['max_depth', 'max_features']
        elif model_name == 'figs':
            model = FIGSRegressor(max_rules=args.max_rules, max_trees=args.max_trees, max_features=args.max_features)
            params = ['max_rules', 'max_trees','max_features']
        elif model_name == 'xgboost':
            model = xgb.XGBRegressor(max_depth=args.max_depth)
            #model = xgb.XGBRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)
            params = ['max_depth']
        elif model_name == 'xgboost30':
            model = xgb.XGBRegressor(n_estimators=30, max_depth=args.max_depth)
            params = ['max_depth']
        elif model_name == 'multi_subset':
            model = MultiOutputRegressor(L0L2RegressorCV(max_support_size = self.post_max_features, cv=self.cv))
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
                 post_interaction=args.post_interaction, post_max_features=args.post_max_features, size_interactions=args.max_depth, mo=args.mo)
            params = ['pre_interaction', 'pre_max_features', 'post_interaction', 'post_max_features']
        else:
            model = None
        return model
    elif task_type in ['binary', 'multiclass', 'classification']:
        if model_name == 'FIGSClassifier':
            model = FIGSClassifier(max_rules=args.max_rules, max_trees=args.max_trees, max_depth=args.max_depth)
        elif model_name == 'log_reg':
            model = LogisticRegressionCV(cv=3, penalty='elasticnet', solver='newton-cg', max_iter=100)
        elif model_name == 'dt_binary_mapper':
            model = DTClassBinaryMapper(depth=args.binary_mapper_depth, bit=args.binary_mapper_bit)
        elif model_name == 'gmm_binary_mapper':
            model = GMMBinaryMapper()
            params = []
        elif model_name == 'decision_tree':
            model = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=5)
        elif model_name == 'random_forest':
            model = RandomForestClassifier(max_depth=args.max_depth, min_samples_leaf=5)
        elif model_name == 'rf_plus':
            rf_model = RandomForestClassifier(max_depth=args.max_depth, min_samples_leaf=5, max_features=args.max_features)
            model = RandomForestPlusClassifier(rf_model=rf_model)
        elif model_name == 'figs':
            model = FIGSClassifier(max_rules=args.max_rules, max_trees=args.max_trees, max_features=args.max_features)
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
                 post_interaction=args.post_interaction, post_max_features=args.post_max_features, size_interactions=args.max_depth)
        else:
            model = None
        return model
    else:
        raise ValueError('Invalid task_type: {}'.format(task_type))
        return None