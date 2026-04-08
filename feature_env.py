'''
feature env
interactive with the actor critic for the state and state after action
'''
from collections import namedtuple

from utils.logger import error, info
from utils.tools import feature_state_generation, downstream_task_new, test_task_new, cluster_features

TASK_DICT = {'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary': 'cls',
             'bike_share': 'reg', 'german_credit': 'cls', 'higgs': 'cls',
             'housing_boston': 'reg', 'ionosphere': 'cls', 'lymphography': 'cls',
             'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls',
             'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
             'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
             'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
             'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg'
             }

MEASUREMENT = {
    'cls': ['precision', 'recall', 'f1_score'],
    'reg': ['mae', 'mse', 'rae']
}

REPLAY = {
    'random', 'per'
}

class FeatureEnv:
    def __init__(self, task_name, task_type=None, ablation_mode=''):
        self.task_name = task_name
        if task_type is None:
            self.task_type = TASK_DICT[task_name]
        else:
            self.task_type = task_type
        self.model_performance = namedtuple('ModelPerformance', MEASUREMENT[self.task_type])
        if ablation_mode == '-c':
            self.mode = 'c'
        else:
            self.mode = ''
            
        # --- CACHE THE DOMAIN SPLIT ---
        # Memorize the row order so we don't need column '9' later
        if self.task_name == 'housing_boston':
            import numpy as np
            import os
            base = os.path.dirname(os.path.abspath(__file__))
            idx_path = os.path.join(base, 'data/processed/environments/current_env_index.npy')
            self.env_sort_index = np.load(idx_path)
    '''
        input a Dataframe (cluster or feature set)
        :return the feature status
        return type is Numpy array
    '''
    def get_feature_state(self, data):
        return feature_state_generation(data)

    '''
        input a Dataframe (cluster or feature set)
        :return the current dataframe performance
        return type is Numpy array
    '''
    def get_reward(self, data):
        if self.task_name == 'housing_boston':
            import numpy as np
            
            # Sort data using the cached row order (bypasses missing '9')
            sorted_data = data.loc[self.env_sort_index].copy()
            
            # --- THE ULTIMATE ANTI-CRASH SANITIZER ---
            # 1. Catch explicit Infinities
            sorted_data = sorted_data.replace([np.inf, -np.inf], np.nan)
            # 2. Clip astronomical numbers that overflow sklearn's float32 limit
            sorted_data = sorted_data.clip(lower=-1e15, upper=1e15)
            # 3. Fill any NaNs (from infs or invalid math like log of negatives) with 0
            sorted_data = sorted_data.fillna(0)
            
            total_rows = len(sorted_data)
            split_1 = int(total_rows * 0.35)
            split_2 = int(total_rows * 0.65)

            env_A = sorted_data.iloc[:split_1].reset_index(drop=True)
            env_B = sorted_data.iloc[split_2:].reset_index(drop=True)
            
            score_A = downstream_task_new(env_A, self.task_type, 'generic')
            score_B = downstream_task_new(env_B, self.task_type, 'generic')
            
            # 2. Base Math
            mean_score = (score_A + score_B) / 2.0
            variance = abs(score_A - score_B)
            
            strict_lambda = 2.0 
            
            irm_reward = mean_score - (strict_lambda * variance)
            return irm_reward
        else:
            return downstream_task_new(data, self.task_type, self.task_name)

    '''
        input a Dataframe (cluster or feature set)
        :return the current dataframe performance on few dataset
        its related measure is listed in {MEASUREMENT[self.task_type]}
        return type is Numpy array
    '''
    def get_performance(self, data):
        if self.task_name == 'housing_boston':
            import numpy as np
            
            # THE ULTIMATE BLIND TEST: Evaluate ONLY on Env C (Middle Tax)
            # Sort data using the cached row order
            sorted_data = data.loc[self.env_sort_index].copy()
            
            # --- THE ULTIMATE ANTI-CRASH SANITIZER ---
            # 1. Catch explicit Infinities
            sorted_data = sorted_data.replace([np.inf, -np.inf], np.nan)
            # 2. Clip astronomical numbers that overflow sklearn's float32 limit
            sorted_data = sorted_data.clip(lower=-1e15, upper=1e15)
            # 3. Fill any NaNs (from infs or invalid math like log of negatives) with 0
            sorted_data = sorted_data.fillna(0)
            
            total_rows = len(sorted_data)
            split_1 = int(total_rows * 0.35)
            split_2 = int(total_rows * 0.65)

            env_C = sorted_data.iloc[split_1:split_2].reset_index(drop=True)
            # Test the final features on a domain the RL agent has NEVER seen
            a, b, c = test_task_new(env_C, task=self.task_type, task_name='generic')
        else:
            a, b, c = test_task_new(data, task=self.task_type, task_name=self.task_name)
            
        return self.model_performance(a, b, c)

    def cluster_build(self, X, y, cluster_num):
        return cluster_features(X, y, cluster_num, mode=self.mode)

    def report_performance(self, original, opt):
        report = self.get_performance(opt)
        original_report = self.get_performance(original)
        info('Original Feature Count: {}, Generated Feature Count: {}'.format(original.shape[1], opt.shape[1]))
        if self.task_type == 'reg':
            final_result = report.rae
            info('MAE on original is: {:.3f}, MAE on generated is: {:.3f}'.
                 format(original_report.mae, report.mae))
            info('RMSE on original is: {:.3f}, RMSE on generated is: {:.3f}'.
                 format(original_report.mse, report.mse))
            info('1-RAE on original is: {:.3f}, 1-RAE on generated is: {:.3f}'.
                 format(original_report.rae, report.rae))
        elif self.task_type == 'cls':
            final_result = report.f1_score
            info('Pre on original is: {:.3f}, Pre on generated is: {:.3f}'.
                 format(original_report.precision, report.precision))
            info('Rec on original is: {:.3f}, Rec on generated is: {:.3f}'.
                 format(original_report.recall, report.recall))
            info('F-1 on original is: {:.3f}, F-1 on generated is: {:.3f}'.
                 format(original_report.f1_score, report.f1_score))
        elif self.task_type == 'det':
            final_result = report.ras
            info(
                'Average Precision Score on original is: {:.3f}, Average Precision Score on generated is: {:.3f}'
                .format(original_report.map, report.map))
            info(
                'F1 Score on original is: {:.3f}, F1 Score on generated is: {:.3f}'
                .format(original_report.f1_score, report.f1_score))
            info(
                'ROC AUC Score on original is: {:.3f}, ROC AUC Score on generated is: {:.3f}'
                .format(original_report.ras, report.ras))
        else:
            error('wrong task name!!!!!')
            assert False
        return final_result
# class FeatureEnv(Env):
