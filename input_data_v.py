#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:42:32 2019

@author: shixiaohan
"""

import pickle
import numpy as np 
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import data_process
import models
import train_eval
import options

with open('11.pickle', 'rb') as file:
    train_map =pickle.load(file)
    
step = 6

org_train_map = []
for i in range(len(train_map)):
    org_train_map1 = []
    for x in range(len(train_map[i])):       
        org_train_map1.append(train_map[i][x])
    org_train_map.append(org_train_map1)

trainlabel = []
for i in range(len(org_train_map)):
    input_trainlabel_1 = []
    for x in range(len(org_train_map[i])):
        input_trainlabel_1_1 = []
        input_trainlabel_1_1.append(org_train_map[i][x]['emotion_v'])
        input_trainlabel_1.append(input_trainlabel_1_1) 
    trainlabel.append(input_trainlabel_1)  

  
input_trainlabel = []
for i in range(len(trainlabel)):
    input_trainlabel_1 = []
    for x in range(len(trainlabel[i]) - step):
        a = trainlabel[i][x+step][0]
        if(a <= 2):
            a = 0
        if(2 < a < 4):
            a = 1
        if(a >= 2):
            a = 2
        input_trainlabel_1.append(a) 
    input_trainlabel.append(input_trainlabel_1)                       

                   
input_traindata = [] 
input_traindata_x = []
input_traindata_y = []
input_traindata_z = []
for i in range(len(org_train_map)):
    input_traindata_1 = []  
    input_traindata_2 = []  
    input_traindata_3 = []
    input_traindata_4 = []
    for x in range(len(org_train_map[i]) - step):
        input_train_data_1 = []
        input_train_transcription_1 = []
        input_train_trad_1 = []
        input_train_name_1 = []
        for y in range (step):
            a = org_train_map[i][x+y]['spec_data']
            b = org_train_map[i][x+y]['transcr_data']
            c = org_train_map[i][x+y]['trad_data']
            d = org_train_map[i][x+y]['id']
            input_train_data_1.append(a)
            input_train_transcription_1.append(b)
            input_train_trad_1.append(c)
            input_train_name_1.append(d)            
        input_traindata_1.append(input_train_data_1)
        input_traindata_2.append(input_train_transcription_1)
        input_traindata_3.append(input_train_trad_1) 
        input_traindata_4.append(input_train_name_1)         
    input_traindata.append(input_traindata_1) 
    input_traindata_x.append(input_traindata_2) 
    input_traindata_y.append(input_traindata_3)    
    input_traindata_z.append(input_traindata_4)    
traindata_1_1 = []
for i in range(len(input_traindata)):
    input_traindata_1_1 = []
    for x in range(len(input_traindata[i])):
        a = {}
        a['spec_data'] = input_traindata[i][x]
        a['label'] = input_trainlabel[i][x]
        a['transcr_data'] = input_traindata_x[i][x]
        a['trad_data'] = input_traindata_y[i][x]
        a['id'] = input_traindata_z[i][x]
        input_traindata_1_1.append(a)    
    traindata_1_1.append(input_traindata_1_1)
traindata_1 = []
for i in range(len(traindata_1_1)):
    input_traindata_1_1 = []
    for x in range(len(traindata_1_1[i])):
        if isinstance(traindata_1_1[i][x]['label'],int):
            input_traindata_1_1.append(traindata_1_1[i][x])
    traindata_1.append(input_traindata_1_1)

data_1 = []
data_2 = []
data_3 = []
data_4 = []
data_5 = []
for i in range(len(traindata_1)):
    if(traindata_1[i][0]['id'][0][4] == '1'):
        data_1.append(traindata_1[i])
    if(traindata_1[i][0]['id'][0][4] == '2'):
        data_2.append(traindata_1[i])
    if(traindata_1[i][0]['id'][0][4] == '3'):
        data_3.append(traindata_1[i])
    if(traindata_1[i][0]['id'][0][4] == '4'):
        data_4.append(traindata_1[i])
    if(traindata_1[i][0]['id'][0][4] == '5'):
        data_5.append(traindata_1[i])
data = []
data.append(data_1)
data.append(data_2)
data.append(data_3)
data.append(data_4)
data.append(data_5)


final_result =[]
final_f1 =[]

kf = KFold(n_splits=5)
for index,(train,test) in enumerate(kf.split(data)):
    train_data = []
    test_data = []
    for i in range(len(train)):
        train_data.extend(data[train[i]])
    for i in range(len(test)):
        test_data.extend(data[test[i]])
    input_train_data_spec = []
    for i in range(len(train_data)):
        for x in range(len(train_data[i])):
            input_train_data_spec.append(train_data[i][x]['spec_data'])    
    input_train_data_trad = []
    for i in range(len(train_data)):
        for x in range(len(train_data[i])):
            input_train_data_trad.append(train_data[i][x]['trad_data']) 
    input_train_data_trad = np.array(input_train_data_trad)
    input_train_data_tran = []
    for i in range(len(train_data)):
        for x in range(len(train_data[i])):
            input_train_data_tran.append(train_data[i][x]['transcr_data'])             
    input_train_data_tran = np.array(input_train_data_tran)     
    input_train_label = []
    for i in range(len(train_data)):
        for x in range(len(train_data[i])):
            input_train_label.append(train_data[i][x]['label'])

    input_test_data_id = []
    for i in range(len(test_data)):
        input_test_data_id.append(test_data[i][0]['id'][0][0:-5])  
             
    input_test_data_spec = []
    for i in range(len(test_data)):
        for x in range(len(test_data[i])):
            input_test_data_spec.append(test_data[i][x]['spec_data'])       
    input_test_data_trad = []
    for i in range(len(test_data)):
        for x in range(len(test_data[i])):
            input_test_data_trad.append(test_data[i][x]['trad_data'])    
    input_test_data_trad = np.array(input_test_data_trad)  
    input_test_data_tran = []
    for i in range(len(test_data)):
        for x in range(len(test_data[i])):        
            input_test_data_tran.append(test_data[i][x]['transcr_data'])  
    input_test_data_tran = np.array(input_test_data_tran)    
    input_test_label = []
    for i in range(len(test_data)):
        for x in range(len(test_data[i])):
            input_test_label.append(test_data[i][x]['label'])
    input_test_orgin_label = []
    for i in range(len(test_data)):
        label = []
        for x in range(len(test_data[i])):
            label.append(test_data[i][x]['label'])
        input_test_orgin_label.append(label)
            
    
    data_getter = data_process.data_get(input_train_data_spec,input_train_label)
    label = data_getter.get_tar()
    label = np.array(label,dtype='int64')
    indata = data_getter.get_multi_frames(frames=5) 

    data_getter=data_process.data_get(input_test_data_spec,input_test_label)
    label_test=data_getter.get_tar()
    label_test=np.array(label_test,dtype='int64')
    indata_test=data_getter.get_multi_frames(frames=5)        
    
    HIDDEN_SIZE=512
    HIDDEN_SIZE_dio=512
    HIDDEN_SIZE_output=512    
    OUTPUT_SIZE=3
    
    Options=options.options_setting(utt_layers=1,
                                    dia_layers=1,
                                    output_layers=1,
                                    lr = 0.0001,
                                    epoch = 20,
                                    dropout = 0.2,
                                    batch_size = 1,
                                    print_every=500,
                                    bid_flag = False,
                                    dia_bid_flag=False)
    
    utt_insize=len(indata[0][0][0])
    dia_insize=Options.dia_layers*HIDDEN_SIZE + 768 + 88
    output_insize = Options.dia_layers*HIDDEN_SIZE + 768 + 88
    
    utt_net = models.Utterance_net(utt_insize,HIDDEN_SIZE,Options)
    dia_net_a =models.Dialogue_net(dia_insize,HIDDEN_SIZE_dio,OUTPUT_SIZE,Options)
    dia_net_b = models.Dialogue_net(dia_insize,HIDDEN_SIZE_dio,OUTPUT_SIZE,Options)    
    output_net = models.Output_net(output_insize,HIDDEN_SIZE_output,OUTPUT_SIZE,Options)
    line_net = models.Line_net(1024,512,OUTPUT_SIZE,Options)
    
    if(Options.GPU_USE):
        utt_net = utt_net.cuda()
        dia_net_a = dia_net_a.cuda()
        dia_net_b = dia_net_b.cuda()        
        output_net = output_net.cuda()
        line_net = line_net.cuda()
        
    exprocess=train_eval.experiment_process()
    pre_label,f1 =exprocess.train_ite(indata,input_train_data_tran,input_train_data_trad,label,
                                      indata_test,input_test_data_tran,input_test_data_trad,label_test,
                                      utt_net,dia_net_a,dia_net_b,output_net,line_net,Options,input_test_orgin_label)

    print(index)
    print("*"*20)
    
    onegroup_result = []
    for i in range(len(input_test_data_id)):
        a = {}      
        a['id'] = input_test_data_id[i]
        a['predict_label'] = pre_label[i]
        a['true_label'] = input_test_orgin_label[i]

        onegroup_result.append(a)
    final_result.append(onegroup_result)
    final_f1.append(f1)

file = open('final_result_v_new.pickle', 'wb')
#file = open('final_result_a_new.pickle', 'wb')
pickle.dump(final_result, file)
file.close()    

file = open('final_f1.pickle', 'wb')
pickle.dump(final_f1, file)
file.close()