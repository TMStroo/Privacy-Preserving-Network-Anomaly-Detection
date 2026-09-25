#!/usr/bin/env python
"""Create minimal sample UNSW-NB15 data for CI testing."""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 100

df = pd.DataFrame({
    'srcip': ['1.1.1.1'] * n,
    'dstip': ['2.2.2.2'] * n,
    'sport': [80] * n,
    'dsport': [443] * n,
    'proto': ['tcp'] * n,
    'state': ['CON'] * n,
    'dur': np.random.exponential(10, n),
    'sbytes': np.random.randint(100, 1000, n),
    'dbytes': np.random.randint(100, 1000, n),
    'sttl': [64] * n,
    'dttl': [64] * n,
    'sloss': [0] * n,
    'dloss': [0] * n,
    'service': ['http'] * n,
    'sload': [1000] * n,
    'dload': [1000] * n,
    'spkts': [10] * n,
    'dpkts': [10] * n,
    'swin': [65535] * n,
    'dwin': [65535] * n,
    'stcpb': [0] * n,
    'dtcpb': [0] * n,
    'tcprtt': [0.1] * n,
    'synack': [0.05] * n,
    'ackdat': [0.05] * n,
    'smean': [100] * n,
    'dmean': [100] * n,
    'trans_depth': [1] * n,
    'response_body_len': [0] * n,
    'ct_srv_src': [1] * n,
    'ct_state_ttl': [1] * n,
    'ct_dst_ltm': [1] * n,
    'ct_src_dport_ltm': [1] * n,
    'ct_dst_sport_ltm': [1] * n,
    'ct_dst_src_ltm': [1] * n,
    'is_ftp_login': [0] * n,
    'ct_ftp_cmd': [0] * n,
    'ct_flw_http_mthd': [1] * n,
    'ct_src_ltm': [1] * n,
    'ct_srv_dst': [1] * n,
    'is_sm_ips_ports': [0] * n,
    'attack_cat': ['Normal'] * n,
    'label': np.random.randint(0, 2, n)
})

df.to_csv('data/raw/UNSW_NB15_training-set.csv', index=False)
df.to_csv('data/raw/UNSW_NB15_testing-set.csv', index=False)

# Features file (minimal)
features_df = pd.DataFrame({
    'Name': df.columns.tolist(),
    'Type': ['nominal' if c in ['proto','service','state','attack_cat'] else 'integer' if df[c].dtype in ['int64','int32'] else 'float' for c in df.columns],
    'Description': ['Feature'] * len(df.columns)
})
features_df.to_csv('data/raw/UNSW_NB15_features.csv', index=False)

print('Sample data created successfully')
print(f'Train: {len(df)} rows, Test: {len(df)} rows')