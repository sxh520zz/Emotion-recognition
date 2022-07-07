import os
import time
import random
import argparse
import pickle
import copy
import torch
import numpy as np
import torch.utils.data as Data
import torch.utils.data.dataset as Dataset
from sklearn import preprocessing
import torch.optim as optim
from torch.autograd import Variable
from models import Utterance_net
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold


class subDataset(Dataset.Dataset):
    def __init__(self,Data_1,Label):
        self.Data_1 = Data_1
        self.Label = Label
    def __len__(self):
        return len(self.Data_1)
    def __getitem__(self, item):
        data_1 = self.Data_1[item]
        data_2 = []
        data_3 = []
        for i in range(len(data_1)):
            if(i % 2 == 0):
                data_2.append(data_1[i])
            else:
                data_3.append(data_1[i])
        data_1 = torch.Tensor(data_1)
        data_2 = torch.Tensor(data_2[-1])
        data_3 = torch.Tensor(data_3[-1])
        label = torch.Tensor(self.Label[item])
        return data_1,data_2,data_3,label

def Feature(data):
    input_train_data_trad = []
    for i in range(len(data)):
        input_train_data_trad.append(data[i]['trad_data'])
    input_label = []
    for i in range(len(data)):
        input_label.append(data[i]['label'])
    input_data_id= []
    for i in range(len(data)):
        input_data_id.append(data[i]['id'][0][0:-5])
    input_orgin_label = []
    for i in range(len(data)):
        input_orgin_label.append(data[i]['label'])
    return input_train_data_trad,input_label,input_data_id,input_orgin_label

def Get_data(data,train,test,args):
    train_data = []
    test_data = []
    for i in range(len(train)):
        train_data.extend(data[train[i]])
    for i in range(len(test)):
        test_data.extend(data[test[i]])
    w = len(test_data)
    input_train_data_trad,input_train_label,_,_ = Feature(train_data)
    input_test_data_trad,input_test_label,input_test_data_id,input_test_label_org = Feature(test_data)
    print(len(input_train_label))
    print(len(input_test_label))
    label = np.array(input_train_label).reshape(-1, 1)
    label_test = np.array(input_test_label).reshape(-1,1)
    train_dataset = subDataset(input_train_data_trad,label)
    test_dataset = subDataset(input_test_data_trad,label_test)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,drop_last=False, shuffle=False)
    return train_loader,test_loader,input_test_data_id,input_test_label_org,w