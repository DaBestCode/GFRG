import pandas as pd
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler

VALID_ENVS = ['TAX_TAIL', 'SOCIO_TAIL', 'COMPOUND_TAIL']

if len(sys.argv) < 2 or sys.argv[1].upper() not in VALID_ENVS:
    print(f"Usage: python split_environments.py [ENV_NAME]")
    print(f"Available envs: {VALID_ENVS}")
    sys.exit(1)

env_name = sys.argv[1].upper()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data/processed/housing_boston.hdf')
df = pd.read_hdf(data_path)

out_dir = os.path.join(BASE_DIR, 'data/processed/environments')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


if env_name == 'TAX_TAIL':
    # Env 1: Economic Shift (Call column name 10 directly)
    df['sort_metric'] = df[10] 

elif env_name == 'SOCIO_TAIL':
    # Env 2: Demographic Shift (Call column name 13 directly)
    df['sort_metric'] = df[13]

elif env_name == 'COMPOUND_TAIL':
    # Env 3: Multivariate Euclidean Distance (Call columns 10 and 1 directly)
    scaler = StandardScaler()
    scaled_vars = scaler.fit_transform(df[[10, 1]])
    distances = np.sqrt(scaled_vars[:, 0]**2 + scaled_vars[:, 1]**2)
    df['sort_metric'] = distances

# Sort the dataframe by the calculated metric
df_sorted = df.sort_values(by='sort_metric')
df_final = df_sorted.drop(columns=['sort_metric'])
total_rows = len(df_final)
split_1 = int(total_rows * 0.35)
split_2 = int(total_rows * 0.65)

np.save(os.path.join(out_dir, 'current_env_index.npy'), df_final.index.to_numpy())
env_train_A = df_final.iloc[:split_1].copy()
env_train_B = df_final.iloc[split_2:].copy()
env_test_C = df_final.iloc[split_1:split_2].copy()

env_train_A.to_hdf(os.path.join(out_dir, 'boston_env_A_train.hdf'), key='df', mode='w')
env_train_B.to_hdf(os.path.join(out_dir, 'boston_env_B_train.hdf'), key='df', mode='w')
env_test_C.to_hdf(os.path.join(out_dir, 'boston_env_C_test.hdf'), key='df', mode='w')

print(f"Successfully created 3 robust Domains for {env_name}!")