import os
import re
import csv
import operator
import pickle

dir = os.getcwd()
label_dir = "OpenSmile/IEMOCAP_9946.csv"
Smile_dir = "OpenSmile/isGeMAPs_iemocap9946.csv"
rootdir = dir + '/IEMOCAP_full_release'

step = 6

def as_num(x):
    y = '{:.5f}'.format(x)  # .10f 保留10位小数
    return y
def emo_change(x):
    if x == 'xxx' or x == 'oth':
        x = 0
    if x == 'neu':
        x = 1
    if x == 'hap':
        x = 2
    if x == 'ang':
        x = 3
    if x == 'sad':
        x = 4
    if x == 'exc':
        x = 5
    if x == 'sur':
        x = 6
    if x == 'fea':
        x = 7
    if x == 'dis':
        x = 8
    if x == 'fru':
        x = 9
    return x
def load_ALL_DATA():
    label_list= [1,2,3,4,5,6,7,8,9,0]
    id_data = []
    id_label = []
    traindata_map_1 = []
    traindata_map_2 = []
    train_data_map = []

    file = open(Smile_dir, 'r')
    file_content = csv.reader(file)
    for row in file_content:
        if (row[0][0] == 'S'):
            data = {}
            data['id'] = row[0]
            data_1 = row[1:]
            a = []
            for i in range(len(data_1)):
                if ('E' in data_1[i] or 'e' in data_1[i]):
                    x = as_num(float(data_1[i]))
                    a.append(float(x))
                else:
                    a.append(float(x))
            data['Data'] = a
            id_data.append(data)

    for speaker in os.listdir(rootdir):
        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
            text_dir = rootdir + '/' + speaker +  '/dialog/transcriptions'
            for sess in os.listdir(text_dir):
                if (sess[7] in ['i','s']):
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
            emoevl = rootdir + '/' + speaker +  '/dialog/EmoEvaluation'
            for sess in os.listdir(emoevl):
                if (sess[2] in ['s']):
                    data_map2_1 = []
                    emotdir = emoevl + '/' + sess
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
                                emot_map['label_cat'] = emo_change(t[4])
                                if (emot_map['label_cat'] in label_list):
                                    #if (emot_map['label_cat'] == 5):
                                        #emot_map['label_cat'] = 2
                                    #emot_map['label_cat'] = emot_map['label_cat'] - 1
                                    a = emot_map.copy()
                                    data_map2_1.append(a)
                    traindata_map_2.append(data_map2_1)
    print(len(traindata_map_2))
    print(len(traindata_map_2[0]))
    for i in range(len(traindata_map_1)):
        for x in range(len(traindata_map_1[i])):
            for j in range(len(traindata_map_2)):
                for y in range(len(traindata_map_2[j])):
                    if (operator.eq(traindata_map_1[i][x]['id'], traindata_map_2[j][y]['id'])):
                        a = traindata_map_2[j][y].copy()
                        traindata_map_1[i][x].update(a)
                        break
    for i in range(len(traindata_map_1)):
        for x in range(len(traindata_map_1[i])):
            for j in range(len(id_data)):
                if(traindata_map_1[i][x]['id'] == id_data[j]['id']):
                    traindata_map_1[i][x]['trad_data'] = id_data[j]['Data']
    for i in range(len(traindata_map_1)):
        data_map_1 = []
        for x in range(len(traindata_map_1[i])):
            if (len(traindata_map_1[i][x]) == 7):
                data_map_1.append(traindata_map_1[i][x])
        train_data_map.append(data_map_1)
    '''
    speaker = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    data = [[], [], [], [], [], [], [], [], [], []]
    for i in range(len(all_data)):
        for j in range(len(speaker)):
            if (all_data[i]['speaker'] == speaker[j]):
                data[j].append(all_data[i])  
    '''

    return train_data_map
def Rebuild_data(train_data_map):
    for i in range(len(train_data_map)):
        for x in range(len(train_data_map[i]) - 1):
            if (train_data_map[i][x]['id'][-4] == train_data_map[i][x + 1]['id'][-4]):
                train_data_map[i][x + 1]['transcription'] = train_data_map[i][x]['transcription'] + \
                                                            train_data_map[i][x + 1]['transcription']
                train_data_map[i][x + 1]['emotion_a'] = (train_data_map[i][x]['emotion_a'] + train_data_map[i][x + 1][ 'emotion_a']) / 2
                train_data_map[i][x + 1]['emotion_v'] = (train_data_map[i][x]['emotion_v'] + train_data_map[i][x + 1]['emotion_v']) / 2
                train_data_map[i][x + 1]['emotion_d'] = (train_data_map[i][x]['emotion_d'] + train_data_map[i][x + 1]['emotion_d']) / 2
                train_data_map[i][x + 1]['label_cat'] = train_data_map[i][x + 1]['label_cat']
                #train_data_map[i][x + 1]['spec_data'] = np.vstack((train_data_map[i][x]['spec_data'], train_data_map[i][x + 1]['spec_data']))
                for w in range(len(train_data_map[i][x]['trad_data'])):
                    train_data_map[i][x + 1]['trad_data'][w] = (train_data_map[i][x]['trad_data'][w] +
                                                                train_data_map[i][x + 1]['trad_data'][w]) / 2
                train_data_map[i][x].clear()

    train_data_map_1 = []
    for i in range(len(train_data_map)):
        data_map_1 = []
        for x in range(len(train_data_map[i])):
            if (len(train_data_map[i][x]) == 7):
                data_map_1.append(train_data_map[i][x])
        train_data_map_1.append(data_map_1)
    return train_data_map_1

def Train_data(train_map):
    label_list= [1,2,3,4,5]
    input_traindata_x = []
    input_traindata_y = []
    input_traindata_z = []
    for i in range(len(train_map)):
        input_trainlabel_2 = []
        input_traindata_3 = []
        input_traindata_4 = []
        for x in range(len(train_map[i]) - step):
            input_train_trad_1 = []
            input_train_name_1 = []
            for y in range(step):
                c = train_map[i][x + y]['trad_data']
                d = train_map[i][x + y]['id']
                input_train_trad_1.append(c)
                input_train_name_1.append(d)
            input_trainlabel_2.append(train_map[i][x + step]['label_cat'])
            input_traindata_3.append(input_train_trad_1)
            input_traindata_4.append(input_train_name_1)
        input_traindata_x.append(input_trainlabel_2)
        input_traindata_y.append(input_traindata_3)
        input_traindata_z.append(input_traindata_4)
    num = 0
    traindata_1 = []
    for i in range(len(input_traindata_z)):
        input_traindata_1_1 = []
        for x in range(len(input_traindata_z[i])):
            a = {}
            if (input_traindata_x[i][x] in label_list):
                if (input_traindata_x[i][x] == 5):
                    a['label'] = 2
                a['label'] = input_traindata_x[i][x] - 1
                a['trad_data'] = input_traindata_y[i][x]
                a['id'] = input_traindata_z[i][x]
                input_traindata_1_1.append(a)
                num = num + 1
        traindata_1.append(input_traindata_1_1)
    print(num)
    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    data_5 = []
    for i in range(len(traindata_1)):
        if (traindata_1[i][0]['id'][0][4] == '1'):
            data_1.append(traindata_1[i])
        if (traindata_1[i][0]['id'][0][4] == '2'):
            data_2.append(traindata_1[i])
        if (traindata_1[i][0]['id'][0][4] == '3'):
            data_3.append(traindata_1[i])
        if (traindata_1[i][0]['id'][0][4] == '4'):
            data_4.append(traindata_1[i])
        if (traindata_1[i][0]['id'][0][4] == '5'):
            data_5.append(traindata_1[i])
    data = []
    data.append(data_1)
    data.append(data_2)
    data.append(data_3)
    data.append(data_4)
    data.append(data_5)
    return data

ALL_DATA = load_ALL_DATA()
Re_data = Rebuild_data(ALL_DATA)
Train_data = Train_data(Re_data)

file = open('Train_data.pickle', 'wb')
pickle.dump(Train_data, file)
file.close()