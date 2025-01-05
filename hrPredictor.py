# import sys
# import joblib
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from scipy.signal import butter, filtfilt
#
# physio_model = joblib.load('physio_model.pkl')
#
# def denoise_signal(signal, fs=100.0, cutoff=40.0):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(1, normal_cutoff, btype='low', analog=False)
#     return filtfilt(b, a, signal)
#
# eda_value = float(sys.argv[1])  # EDA value from Node.js
# hr_value = float(sys.argv[2])   # Heart rate value from Node.js
#
# input_data = pd.DataFrame({'HR': [hr_value], 'EDA': [eda_value]})
#
# input_data['HR'] = input_data['HR'].clip(lower=50, upper=180)
# input_data['EDA'] = input_data['EDA'].clip(lower=0, upper=6)
#
# scaler = MinMaxScaler()
# input_data[['HR', 'EDA']] = scaler.fit_transform(input_data[['HR', 'EDA']])
#
# for col in ['EDA', 'HR']:
#     input_data[col] = denoise_signal(input_data[col], fs=100.0)
#
# prediction = physio_model.predict(input_data)[0]
#
# print(prediction)
import sys
import joblib
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt
import numpy as np

physio_model = joblib.load('physio_model.pkl')

def denoise_signal(signal, fs=100.0, cutoff=40.0):
    if len(signal) <= 6:  
            # signal = np.pad(signal, (6 - len(signal), 0), mode='edge')
        return signal
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

eda_value = float(sys.argv[1]) 
hr_value = float(sys.argv[2]) 

input_data = pd.DataFrame({'HR': [hr_value], 'EDA': [eda_value]})
input_data['HR'] = input_data['HR'].clip(lower=50, upper=180)
input_data['EDA'] = input_data['EDA'].clip(lower=0, upper=6)

scaler = MinMaxScaler()
input_data[['HR', 'EDA']] = scaler.fit_transform(input_data[['HR', 'EDA']])

for col in ['EDA', 'HR']:
    input_data[col] = denoise_signal(input_data[col], fs=100.0)

probability = physio_model.predict_proba(input_data)[0][1]

output = {"eda": eda_value, "hr": hr_value, "probability": probability}
print(json.dumps(output))
