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


label_path = '/media/bagus/data01/dataset/IEMOCAP_full_release/data_collected_full_10036.pickle'
vader_path = '/media/bagus/data01/github/text-vad/VADanalysis/out/vader/Output_Vader_Sentiment_iemocap_text_10036.csv'

with open(label_path, 'rb') as handle:
    data = pickle.load(handle)
v_true = np.array([v['v'] for v in data]).reshape(10036, 1)

# output vader
vader_output = pd.read_csv(vader_path, sep=';')
vader_pred = np.array(vader_output['Sentiment']).reshape(10036, 1)

# do normalization
# scale label score in range (-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))

scaler.fit(v_true)
v_true = scaler.transform(v_true)

# calculate CCC
v_mae = mean_absolute_error(v_true, vader_pred)
v_mse = mean_squared_error(v_true, vader_pred)
v_ccc_vader = CCC_numpy(v_true.flatten(), vader_pred.flatten())
print('MAE, MSE, CCC: ', v_mae, v_mse, v_ccc_vader)
