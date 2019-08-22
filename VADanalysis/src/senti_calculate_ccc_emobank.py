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

## emobank
label_path = '/media/bagus/data01/s3/course/minor/dimensional_emotion/emobank.csv'
pred_path = '/media/bagus/data01/github/text-vad/VADanalysis/out/senti_mika/emobank_te.csv'

data = pd.read_csv(label_path, sep=',')
v_true = np.array(data['V']).reshape(10062,1)

# read prediction score
output = pd.read_csv(pred_path, header=None)

v_pred = np.array(output)

# scale both true and pred score in range (-1, 1)
scaler = MinMaxScaler(feature_range=(-1,1))

scaler.fit(v_true)
v_true = scaler.transform(v_true)

# calculate error
v_mae = mean_absolute_error(v_true, v_pred)
v_mse = mean_squared_error(v_true, v_pred)
v_ccc = CCC_numpy(v_true.flatten(), v_pred.flatten())
print(v_mae, v_mse, v_ccc)
