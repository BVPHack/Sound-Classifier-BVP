# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 20:18:08 2018

@author: pranay
"""

import pyaudio
import librosa
import time
import numpy as np
#import RingBuffer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
import keras
from keras.models import load_model
from keras import backend as K
import tensorflow as tf



#model._make_predict_function()
global model
model= load_model('voice_classification_model5000v2.h5')
global graph
graph = tf.get_default_graph()

# ring buffer will keep the last 2 seconds worth of audio
#ringBuffer = RingBuffer(2 * 22050)

classes = ["saw", "vehicle", "noise"]
p = pyaudio.PyAudio()
def callback(in_data, frame_count, time_info, flag):
    audio_data = np.fromstring(in_data, dtype=np.float32)
    
    # we trained on audio with a sample rate of 22050 so we need to convert it
    audio_data = librosa.resample(audio_data, 44100, 22050)
    mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=40).T,axis=0) 
    mfccs = np.array(mfccs)
    mfccs = mfccs.reshape((1, 40))
    time.sleep(1)
    with graph.as_default():
        index = np.argmax(model.predict(mfccs))
        print(classes[index])
    #ringBuffer.extend(audio_data)

    # machine learning model takes wavform as input and
    # decides if the last 2 seconds of audio contains a goal
    #if model.is_goal(ringBuffer.get()):
        # GOAL!! Trigger light show
       # requests.get("http://127.0.0.1:8082/goal")
    

    return (in_data, pyaudio.paContinue)

# function that finds the index of the Soundflower
# input device and HDMI output device


stream = p.open(format = pyaudio.paFloat32,
                 channels = 1,
                 rate = 44100,
                 output = False,
                 input = True,
                 stream_callback = callback)

# start the stream
stream.start_stream()

while stream.is_active():
    time.sleep(0.25)

stream.close()
p.terminate()