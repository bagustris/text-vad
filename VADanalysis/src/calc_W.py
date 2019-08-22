# script to calculate average W on ANEW and sentiwordnet
import pandas as pd
import numpy as np

anew = "../lib/EnglishShortened.csv" # ANEW database

#with open(anew, 'rb') as f:
#    data = f.readlines()

data = pd.read_csv(anew, sep=',')
v_mean = np.mean(data['valence'])
print('avg_V: ', v_mean)
print('avg_A: ', np.mean(data['arousal']))
print('avg_D: ', np.mean(data['dominance']))

senti_data = '/media/bagus/data01/s3/course/minor/affect_dictionary/SentiWordNet_3.0.0.txt'

senti_data = pd.read_csv(senti_data, sep='\t')

v_tot = []

for v in range(len(senti_data)):
    v = senti_data.iloc[v,2] - senti_data.iloc[v,3]
    v_tot.append(v)

print('avg_V_senti: ', np.mean(v_tot))
