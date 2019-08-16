# imeocap calculate error and ccc from output of vader and SWN

from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

def CCC_numpy(y_true, y_pred):
    '''Reference numpy implementation of Lin's Concordance correlation coefficient '''
    # covariance between y_true and y_pred
    s_xy = np.cov([y_true, y_pred])[0, 1]
    # means
    x_m = np.mean(y_true)
    y_m = np.mean(y_pred)
    # variances
    s_x_sq = np.var(y_true)
    s_y_sq = np.var(y_pred)
    # condordance correlation coefficient
    ccc = (2.0*s_xy) / (s_x_sq + s_y_sq + (x_m-y_m)**2)   
    return ccc


label_path = '/media/bagus/data01/s3/course/minor/dimensional_emotion/emobank.csv'
pred_path = '/media/bagus/data01/github/text-vad/VADanalysis/src/out_swn_emobank.txt'
vader_path = '/media/bagus/data01/github/text-vad/VADanalysis/out/Output_Vader_Sentiment_emobank_te.csv'

data = pd.read_csv(label_path, sep=',')
v_true = np.array(data['V']).reshape(10062,1)

output = pd.read_csv(pred_path, header=None)
v_pred_swn = np.array(output).reshape(10062, 1)

# output vader
vader_output = pd.read_csv(vader_path, sep=';')
vader_pred = np.array(vader_output['Sentiment']).reshape(10062, 1)

# do normalization
# scale both true and pred score in range (-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))

scaler.fit(v_true)
v_true = scaler.transform(v_true)

scaler.fit(v_pred_swn)
v_pred_swn = scaler.transform(v_pred_swn)

scaler.fit(vader_pred)
vader_pred = scaler.transform(vader_pred)

# calculate CCC
v_ccc_swn = CCC_numpy(v_true.flatten(), v_pred_swn.flatten())
v_ccc_vader = CCC_numpy(v_true.flatten(), vader_pred.flatten())
print('CCC swn: ', v_ccc_swn)
print('CCC vader: ', v_ccc_vader)