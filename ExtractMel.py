#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:32:28 2018

@author: shixiaohan
"""
import re
import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import pickle
import matplotlib.pyplot as plt
import operator
from sklearn import preprocessing


def read_file(filename):
    file = wave.open(filename, 'r')
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype=np.short)
    time = np.arange(0, wav_length) * (1.0 / framerate)
    file.close()
    return wavedata, time, framerate

def read_IEMOCAP():
    train_num = 10039
    filter_num = 40
    rootdir = 'IEMOCAP_full_release'
    train_label = np.empty((train_num, 2), dtype=np.float32)
    train_num = 0
    train_map = {}
    for speaker in os.listdir(rootdir):
        if (speaker[0] == 'S'):
            sub_dir = os.path.join(rootdir, speaker, 'sentences/wav')
            emoevl = os.path.join(rootdir, speaker, 'dialog/EmoEvaluation')
            for sess in os.listdir(sub_dir):
                if (sess[7] in ['i']):
                    emotdir = emoevl + '/' + sess + '.txt'
                    emot_map = {}
                    with open(emotdir, 'r') as emot_to_read:
                        while True:
                            line = emot_to_read.readline()
                            if not line:
                                break
                            if (line[0] == '['):
                                v_a = []
                                t = line.split()
                                x = t[5] + t[6]
                                x = re.split(r'[,[]', x)
                                v_a.append(float(x[1]))
                                v_a.append(float(x[2]))
                                emot_map[t[3]] = v_a
                    file_dir = os.path.join(sub_dir, sess, '*.wav')
                    files = glob.glob(file_dir)
                    for filename in files:
                        wavname = filename.split("/")[-1][:-4]
                        emotion = emot_map[wavname]
                        data, time, rate = read_file(filename)
                        mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
                        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
                            # training set
                            part = mel_spec
                            #delta1 = ps.delta(mel_spec, 2)
                            #delta2 = ps.delta(delta1, 2)
                            #input_data_1 = np.concatenate((part, delta1), axis=1)
                            #input_data = np.concatenate((input_data_1, delta2), axis=1)
                            data1.append(part)
                            train_map['id'] = wavname
                            a = train_map.copy()
                            train_data_map.append(a)
                            train_label[train_num] = emotion
                            train_num = train_num + 1
    for i in range(len(data1)):
        data1[i] = preprocessing.scale(data1[i])

    return data1

def read_IEMOCAP_LABEL():
    traindata_map_1 = []
    traindata_map_2 = []
    train_data_map = []

    rootdir = 'IEMOCAP_full_release'
    for speaker in os.listdir(rootdir):
        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
            text_dir = os.path.join(rootdir, speaker, 'dialog/transcriptions')
            for sess in os.listdir(text_dir):
                if (sess[7] in ['i']):
                    data_map1_1 = []
                    textdir = text_dir + '/' + sess
                    text_map = {}
                    with open(textdir, 'r') as text_to_read:
                        while True:
                            line = text_to_read.readline()
                            if not line:
                                break
                            t = line.split()
                            if (t[0][0] in 'S'):
                                str = " ".join(t[2:])
                                text_map['id'] = t[0]
                                text_map['transcription'] = str
                                a = text_map.copy()
                                data_map1_1.append(a)
                    traindata_map_1.append(data_map1_1)

    for speaker in os.listdir(rootdir):
        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
            emoevl = os.path.join(rootdir, speaker, 'dialog/EmoEvaluation')
            for sess in os.listdir(emoevl):
                if (sess[-1] in ['t']):
                    data_map2_1 = []
                    emotdir = emoevl + '/' + sess
                    # emotfile = open(emotdir)
                    emot_map = {}
                    with open(emotdir, 'r') as emot_to_read:
                        while True:
                            line = emot_to_read.readline()
                            if not line:
                                break
                            if (line[0] == '['):
                                t = line.split()
                                emot_map['id'] = t[3]
                                x = t[5] + t[6] + t[7]
                                x = re.split(r'[,[]', x)
                                y = re.split(r'[]]', x[3])
                                emot_map['emotion_v'] = float(x[1])
                                emot_map['emotion_a'] = float(x[2])
                                emot_map['emotion_d'] = float(y[0])
                                emot_map['label'] = t[4]
                                a = emot_map.copy()
                                data_map2_1.append(a)
                    traindata_map_2.append(data_map2_1)

    for i in range(len(traindata_map_1)):
        for x in range(len(traindata_map_1[i])):
            for j in range(len(traindata_map_2)):
                for y in range(len(traindata_map_2[j])):
                    if (operator.eq(traindata_map_1[i][x]['id'], traindata_map_2[j][y]['id'])):
                        a = traindata_map_2[j][y].copy()
                        traindata_map_1[i][x].update(a)
                        # data_map.append(a)
                        break

    for i in range(len(traindata_map_1)):
        data_map_1 = []
        for x in range(len(traindata_map_1[i])):
            if (len(traindata_map_1[i][x]) == 6):
                data_map_1.append(traindata_map_1[i][x])
        train_data_map.append(data_map_1)

    for i in range(len(train_data_map)):
        data_map_1 = []
        for x in range(len(train_data_map[i])):
            if (len(train_data_map[i][x]) == 6):
                data_map_1.append(train_data_map[i][x])
        train_data_map_1.append(data_map_1)
    return train_data_map,train_data_map_1

def seg_IEMOCAP():
    for i in range(len(train_data_map)):
        for x in range(len(train_data_map[i])):
            for y in range(len(train_data)):
                if (train_data_map[i][x]['id'] == train_map[y]['id']):
                    train_data_map[i][x]['spec_data'] = train_data[y]
    for i in range(len(train_data_map)):
        for x in range(len(train_data_map[i])):
            for y in range(len(Ge_data)):
                if (train_data_map[i][x]['id'] == Ge_data[y]['id']):
                    train_data_map[i][x]['trad_data'] = Ge_data[y]['data']

    # train_data_map = np.array(train_data_map)
    for i in range(len(train_data_map)):
        for x in range(len(train_data_map[i]) - 1):
            if (train_data_map[i][x]['id'][-4] == train_data_map[i][x + 1]['id'][-4]):
                train_data_map[i][x + 1]['transcription'] = train_data_map[i][x]['transcription'] + \
                                                            train_data_map[i][x + 1]['transcription']
                train_data_map[i][x + 1]['emotion_a'] = (train_data_map[i][x]['emotion_a'] + train_data_map[i][x + 1][
                    'emotion_a']) / 2
                train_data_map[i][x + 1]['emotion_v'] = (train_data_map[i][x]['emotion_v'] + train_data_map[i][x + 1][
                    'emotion_v']) / 2
                train_data_map[i][x + 1]['emotion_d'] = (train_data_map[i][x]['emotion_d'] + train_data_map[i][x + 1][
                    'emotion_d']) / 2
                train_data_map[i][x + 1]['label'] = train_data_map[i][x + 1]['label']
                train_data_map[i][x + 1]['spec_data'] = np.vstack(
                    (train_data_map[i][x]['spec_data'], train_data_map[i][x + 1]['spec_data']))
                train_data_map[i][x].clear()
    train_data_map_1 = []
    for i in range(len(train_data_map)):
        data_map_1 = []
        for x in range(len(train_data_map[i])):
            if (len(train_data_map[i][x]) == 7):
                data_map_1.append(train_data_map[i][x])
        train_data_map_1.append(data_map_1)

if __name__ == '__main__':
    train_data_map = []
    train_data_map_1 = []
    data1 = []
    a = {}
    x = 0
    train_data = read_IEMOCAP()
    train_map,traindata_map = read_IEMOCAP_LABEL()
    seg_IEMOCAP()
    #file = open('traindata_map.pickle', 'wb')
    #pickle.dump(train_data_map_1, file)
    #file.close()
