# calculate error between label and predcited by anew

from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

def CCC_numpy(y_true, y_pred):
    '''Reference numpy implementation of Lin's Concordance correlation coefficient'''
    
    # covariance between y_true and y_pred
    s_xy = np.cov([y_true, y_pred])[0,1]
    # means
    x_m = np.mean(y_true)
    y_m = np.mean(y_pred)
    # variances
    s_x_sq = np.var(y_true)
    s_y_sq = np.var(y_pred)
    
    # condordance correlation coefficient
    ccc = (2.0*s_xy) / (s_x_sq + s_y_sq + (x_m-y_m)**2)
    
    return ccc

label_path = '/media/bagus/data01/dataset/IEMOCAP_full_release/data_collected_full_10036.pickle'
pred_path = '/media/bagus/data01/github/text-vad/VADanalysis/out/anew_median/iemocap_text_10036.csv'

with open(label_path, 'rb') as handle:
    data = pickle.load(handle)
    
v_true = np.array([v['v'] for v in data]).reshape(10036,1)
a_true = np.array([a['a'] for a in data]).reshape(10036,1)
d_true = np.array([d['d'] for d in data]).reshape(10036,1)

# read prediction score
output = pd.read_csv(pred_path, sep=';')

v_pred = np.array(output['Valence']).reshape(10036,1)
#v_pred = np.array(output)
a_pred = np.array(output['Arousal']).reshape(10036,1)
d_pred = np.array(output['Dominance']).reshape(10036,1)

# scale both true and pred score in range (-1, 1)
scaler = MinMaxScaler(feature_range=(-1,1))

scaler.fit(v_true)
v_true = scaler.transform(v_true)

scaler.fit(a_true)
a_true = scaler.transform(a_true)

scaler.fit(d_true)
d_true = scaler.transform(d_true)

## for prediction, also change nan to 0
scaler.fit(v_pred)
v_pred = scaler.transform(v_pred)
v_pred = np.nan_to_num(v_pred, nan=0)

scaler.fit(a_pred)
a_pred = scaler.transform(a_pred)
a_pred = np.nan_to_num(a_pred, nan=0)

scaler.fit(d_pred)
d_pred = scaler.transform(d_pred)
d_pred = np.nan_to_num(d_pred, nan=0)

# calculate error
v_mae = mean_absolute_error(v_true, v_pred)
v_mse = mean_squared_error(v_true, v_pred)
v_ccc = CCC_numpy(v_true.flatten(), v_pred.flatten())
print(v_mae, v_mse, v_ccc)

a_mae = mean_absolute_error(a_true, a_pred)
a_mse = mean_squared_error(a_true, a_pred)
a_ccc = CCC_numpy(a_true.flatten(), a_pred.flatten())
print(a_mae, a_mse, a_ccc)

d_mae = mean_absolute_error(d_true, d_pred)
d_mse = mean_squared_error(d_true, d_pred)
d_ccc = CCC_numpy(d_true.flatten(), d_pred.flatten())
print(d_mae, d_mse, d_ccc)
