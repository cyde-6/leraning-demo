# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 18:51:47 2022

@author: 本地账户李浩东
"""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import loadmat
file = r'C:\Users\天网主机\Desktop\拯救者\声源定位与识别2022课程作业\机器学习\机器学习与声信号处理第六次作业\xtrain.mat'
# mat_dtype=True，保证了导入后变量的数据类型与原类型一致。
data = loadmat(file, mat_dtype=True)
sr = 12e3
# 导入后的data是一个字典，取出想要的变量字段即可。
y = data['datatrain']
x,fs = librosa.load('impulse.wav',sr=None)
y = y[2,:]
spec = np.abs(librosa.stft(y, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=np.max)
mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time')
plt.title('Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')
#%%
from __future__ import print_function
from matplotlib import pyplot as plt
# matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
# import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())

