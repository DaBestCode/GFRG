from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from scipy.special import expit
from sklearn import linear_model
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.svm import LinearSVC

from .logger import error, info


def cube(x):
    return x ** 3


def justify_operation_type(o):
    if o == 'sqrt':
        o = np.sqrt
    elif o == 'square':
        o = np.square
    elif o == 'sin':
        o = np.sin
    elif o == 'cos':
        o = np.cos
    elif o == 'tanh':
        o = np.tanh
    elif o == 'reciprocal':
        o = np.reciprocal
    elif o == '+':
        o = np.add
    elif o == '-':
        o = np.subtract
    elif o == '/':
        o = np.divide
    elif o == '*':
        o = np.multiply
    elif o == 'stand_scaler':
        o = StandardScaler()
    elif o == 'minmax_scaler':
        o = MinMaxScaler(feature_range=(-1, 1))
    elif o == 'quan_trans':
        o = QuantileTransformer(random_state=0)
    elif o == 'exp':
        o = np.exp
    elif o == 'cube':
        o = cube
    elif o == 'sigmoid':
        o = expit
    elif o == 'log':
        o = np.log
    else:
        print('Please check your operation!')
    return o


def mi_feature_distance(features, y):
    dis_mat = []
    for i in range(features.shape[1]):
        tmp = []
        for j in range(features.shape[1]):
            tmp.append(np.abs(mutual_info_regression(features[:, i].reshape
                                                     (-1, 1), y) - mutual_info_regression(features[:, j].reshape
                                                                                          (-1, 1), y))[0] / (
                               mutual_info_regression(features[:, i].
                                                      reshape(-1, 1), features[:, j].reshape(-1, 1))[
                                   0] + 1e-05))
        dis_mat.append(np.array(tmp))
    dis_mat = np.array(dis_mat)
    return dis_mat

def invariant_feature_distance(features, y, env_A_idx, env_B_idx):
    """
    Groups features based on their Distribution Shift (MMD) Stability Gradient
    rather than their target correlation.
    """
    dis_mat = np.zeros((features.shape[1], features.shape[1]))
    feature_mmd_scores = []
    
    # 1. Calculate the stability (MMD) of each feature independently
    for i in range(features.shape[1]):
        feature_A = features[env_A_idx, i].reshape(-1, 1)
        feature_B = features[env_B_idx, i].reshape(-1, 1)
        
        # Simple statistical distance (Mean absolute difference) for speed
        # Alternatively, you can plug in the full RBF kernel MMD here
        shift_score = np.abs(np.mean(feature_A) - np.mean(feature_B)) 
        feature_mmd_scores.append(shift_score)
        
    # 2. Group features by how similar their stability is
    for i in range(features.shape[1]):
        for j in range(features.shape[1]):
            # Features with similar shift behavior have distance ~0
            dis_mat[i, j] = np.abs(feature_mmd_scores[i] - feature_mmd_scores[j])
            
    return dis_mat

def feature_distance(feature, y):
    return mi_feature_distance(feature, y)

'''
for ablation study
if mode == c then don't do cluster
'''
def cluster_features(features, y, cluster_num=2, mode=''):
    if mode == 'c':
        return _wocluster_features(features, y, cluster_num)
    else:
        return _cluster_features(features, y, cluster_num)

def _cluster_features(features, y, cluster_num=2):
    k = int(np.sqrt(features.shape[1]))
    total_rows = features.shape[0]
    split_1 = int(total_rows * 0.35)
    split_2 = int(total_rows * 0.65)
    
    env_A_idx = np.arange(split_1)
    env_B_idx = np.arange(split_2, total_rows)
    features = invariant_feature_distance(features, y, env_A_idx, env_B_idx)
    #features = feature_distance(features, y)
    features = features.reshape(features.shape[0], -1)
    clustering = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='single').fit(features)
    labels = clustering.labels_
    clusters = defaultdict(list)
    for ind, item in enumerate(labels):
        clusters[item].append(ind)
    return clusters

'''
return single column as cluster
'''
def _wocluster_features(features, y, cluster_num=2):
    clusters = defaultdict(list)
    for ind, item in enumerate(range(features.shape[1])):
        clusters[item].append(ind)
    return clusters



SUPPORT_STATE_METHOD = {
    'ds'
}


def feature_state_generation(X):
    return _feature_state_generation_des(X)


def _feature_state_generation_des(X):
    feature_matrix = []
    
    
    for i in range(8):
        feature_matrix = feature_matrix + list(X.astype(np.float64).
                                               describe().iloc[i, :].describe().fillna(0).values)
                                               
    
    total_rows = X.shape[0]
    split_1 = int(total_rows * 0.35)
    split_2 = int(total_rows * 0.65)
    
    env_A = X.iloc[:split_1, :]
    env_B = X.iloc[split_2:, :]
    
    
    feature_shift_gaps = np.abs(env_A.mean() - env_B.mean())
    
    
    gap_statistics = list(feature_shift_gaps.describe().fillna(0).values)
    
    # Inject the Domain Gap into the agent's brain
    feature_matrix = feature_matrix + gap_statistics
    
    return feature_matrix




def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(
        y_test) - y_test))
    return error


def downstream_task_new(data, task_type, task_name='housing_boston'):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(int)
    
    if task_name in ['german_credit', 'housing_boston']:
        
        # 1. Dynamically detect the shift just like the Judge does
        total_rows = X.shape[0]
        env_A_sample = X.iloc[:int(total_rows*0.35), :]
        env_B_sample = X.iloc[int(total_rows*0.65):, :]
        
        # We only check the ORIGINAL features (first 13) to define the split,
        # otherwise a generated feature might accidentally become the split column!
        # We only check the ORIGINAL features (first 13) to define the split
        original_feature_count = 13 if task_name == 'housing_boston' else 20
        check_limit = min(original_feature_count, X.shape[1])
        
        # NEW SCALE-INVARIANT MATH
        mean_diff = np.abs(env_A_sample.iloc[:, :check_limit].mean() - env_B_sample.iloc[:, :check_limit].mean())
        std_dev = X.iloc[:, :check_limit].std() + 1e-8
        mmd_per_feature = mean_diff / std_dev
        split_col_name = mmd_per_feature.idxmax()
        
        split_val = X[split_col_name].median()
        train_idx = np.where(X[split_col_name] <= split_val)[0]
        test_idx = np.where(X[split_col_name] > split_val)[0]
        
        if task_name == 'german_credit' or task_type == 'cls':
            clf = RandomForestClassifier(random_state=0)
            clf.fit(X.iloc[train_idx, :], y.iloc[train_idx])
            y_predict = clf.predict(X.iloc[test_idx, :])
            return f1_score(y.iloc[test_idx], y_predict, average='weighted')

        elif task_name == 'housing_boston' or task_type == 'reg':
            reg = RandomForestRegressor(random_state=0)
            reg.fit(X.iloc[train_idx, :], y.iloc[train_idx])
            y_predict = reg.predict(X.iloc[test_idx, :])
            return 1 - relative_absolute_error(y.iloc[test_idx], y_predict)

    # Standard K-Fold Logic for all other datasets
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0)
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    else:
        return -1


def downstream_task(data, task_type, metric_type, state_num=10):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=state_num, shuffle=True)
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        if metric_type == 'acc':
            return accuracy_score(y_test, y_predict)
        elif metric_type == 'pre':
            return precision_score(y_test, y_predict)
        elif metric_type == 'rec':
            return recall_score(y_test, y_predict)
        elif metric_type == 'f1':
            return f1_score(y_test, y_predict, average='weighted')
    if task_type == 'reg':
        reg = RandomForestRegressor(random_state=0).fit(X_train, y_train)
        y_predict = reg.predict(X_test)
        if metric_type == 'mae':
            return mean_absolute_error(y_test, y_predict)
        elif metric_type == 'mse':
            return mean_squared_error(y_test, y_predict)
        elif metric_type == 'rae':
            return 1 - relative_absolute_error(y_test, y_predict)

def insert_generated_feature_to_original_feas(feas, f):
    y_label = pd.DataFrame(feas[feas.columns[len(feas.columns) - 1]])
    y_label.columns = [feas.columns[len(feas.columns) - 1]]
    feas = feas.drop(columns=feas.columns[len(feas.columns) - 1])
    final_data = pd.concat([feas, f, y_label], axis=1)
    return final_data

def downstream_task_cross_validataion(data, task_type):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(int)
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0)
        scores = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted')
        print(scores)
    if task_type == 'reg':
        reg = RandomForestRegressor(random_state=0)
        scores = 1 - cross_val_score(reg, X, y, cv=5, scoring=make_scorer(
            relative_absolute_error))
        print(scores)


def test_task_new(Dg, task='cls', task_name='housing_boston'):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1].astype(int)
    
    if task_name in ['german_credit', 'housing_boston']:
      
        total_rows = X.shape[0]
        env_A_sample = X.iloc[:int(total_rows*0.35), :]
        env_B_sample = X.iloc[int(total_rows*0.65):, :]
        
        # We only check the ORIGINAL features (first 13) to define the split
        original_feature_count = 13 if task_name == 'housing_boston' else 20
        check_limit = min(original_feature_count, X.shape[1])

        # NEW SCALE-INVARIANT MATH
        mean_diff = np.abs(env_A_sample.iloc[:, :check_limit].mean() - env_B_sample.iloc[:, :check_limit].mean())
        std_dev = X.iloc[:, :check_limit].std() + 1e-8
        mmd_per_feature = mean_diff / std_dev
        split_col_name = mmd_per_feature.idxmax()
        
        info(f'>>> DYNAMIC EVAL: Detecting Shift on [{split_col_name}] (MMD: {mmd_per_feature.max():.4f})')
        
        split_val = X[split_col_name].median()
        train_idx = np.where(X[split_col_name] <= split_val)[0]
        test_idx = np.where(X[split_col_name] > split_val)[0]
        
        if task == 'cls':
            clf = RandomForestClassifier(random_state=0)
            clf.fit(X.iloc[train_idx, :], y.iloc[train_idx])
            
            y_pred_train = clf.predict(X.iloc[train_idx, :])
            y_pred_test = clf.predict(X.iloc[test_idx, :])
            
            # Original Metrics
            pre = precision_score(y.iloc[test_idx], y_pred_test, average='weighted')
            rec = recall_score(y.iloc[test_idx], y_pred_test, average='weighted')
            f1_test = f1_score(y.iloc[test_idx], y_pred_test, average='weighted')
            
            # Robustness Metrics
            f1_train = f1_score(y.iloc[train_idx], y_pred_train, average='weighted')
            stability_gap = np.abs(f1_train - f1_test)
            transfer_ratio = f1_test / (f1_train + 1e-6)
            
            info(f'>>> [CLS ROBUSTNESS] Gap: {stability_gap:.4f} | Transfer Ratio: {transfer_ratio:.4f}')
            return pre, rec, f1_test

        elif task == 'reg':
            reg = RandomForestRegressor(random_state=0)
            reg.fit(X.iloc[train_idx, :], y.iloc[train_idx])
            
            y_pred_train = reg.predict(X.iloc[train_idx, :])
            y_pred_test = reg.predict(X.iloc[test_idx, :])
            
            # Original Metrics (1 - Error)
            mae_test = mean_absolute_error(y.iloc[test_idx], y_pred_test)
            mse_test = mean_squared_error(y.iloc[test_idx], y_pred_test)
            rae_test = relative_absolute_error(y.iloc[test_idx], y_pred_test)
            
            # Robustness Metrics
            rae_train = relative_absolute_error(y.iloc[train_idx], y_pred_train)
            stability_gap = np.abs(mean_absolute_error(y.iloc[train_idx], y_pred_train) - mae_test)
            transfer_ratio = (1 - rae_test) / (1 - rae_train + 1e-6)
            p_mmd = np.abs(np.mean(y_pred_train) - np.mean(y_pred_test))
            
            info(f'>>> [REG ROBUSTNESS] Stability Gap: {stability_gap:.4f} | Transfer Ratio: {transfer_ratio:.4f} | P-MMD: {p_mmd:.4f}')
            return 1 - mae_test, 1 - mse_test, 1 - rae_test
    
    # Standard K-Fold Logic for all other datasets
    if task == 'cls':
        clf = RandomForestClassifier(random_state=0)
        pre_list, rec_list, f1_list = [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average=
            'weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted')
                            )
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list)
    elif task == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list = [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(1 - mean_absolute_error(y_test, y_predict))
            mse_list.append(1 - mean_squared_error(y_test, y_predict))
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list)
    else:
        return -1


def overall_feature_selection(best_features, task_type):
    if task_type == 'reg':
        data = pd.concat([fea for fea in best_features], axis=1)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].astype(int)
        reg = linear_model.Lasso(alpha=0.1).fit(X, y)
        model = SelectFromModel(reg, prefit=True)
        X = X.loc[:, model.get_support()]
        new_data = pd.concat([X, y], axis=1)
        mae, mse, rae = test_task_new(new_data, task_type)
        info('mae: {:.3f}, mse: {:.3f}, 1-rae: {:.3f}'.format(mae, mse, 1 -
                                                              rae))
    elif task_type == 'cls':
        data = pd.concat([fea for fea in best_features], axis=1)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].astype(int)
        clf = LinearSVC(C=0.01, penalty='l1', dual=False).fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X = X.loc[:, model.get_support()]
        new_data = pd.concat([X, y], axis=1)
        acc, pre, rec, f1 = test_task_new(new_data, task_type)
        info('acc: {:.3f}, pre: {:.3f}, rec: {:.3f}, f1: {:.3f}'.format(acc,
                                                                        pre, rec, f1))
    return new_data
