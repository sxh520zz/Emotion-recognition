import os
import time
import random
import argparse
import pickle
import copy
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn.utils.rnn as rmm_utils
import torch.utils.data.dataset as Dataset
import torch.optim as optim
from utils import Get_data
from torch.autograd import Variable
from models import Utterance_net,Dialogue_net,Output_net
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = False

with open('Train_data.pickle', 'rb') as file:
    data = pickle.load(file)

parser = argparse.ArgumentParser(description="RNN_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=64, metavar='N')
parser.add_argument('--log_interval', type=int, default=10, metavar='N')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--dia_layers', type=int, default=2)
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=4)
parser.add_argument('--utt_insize', type=int, default=88)
args = parser.parse_args()

torch.manual_seed(args.seed)


def Train(epoch):
    train_loss = 0
    #dia_net_a.train()
    #dia_net_b.train()
    dia_net_all.train()
    #output_net.train()
    for batch_idx, (data_1,data_2,data_3,target) in enumerate(train_loader):
        if args.cuda:
            data_1, data_2,data_3, target = data_1.cuda(), data_2.cuda(),data_3.cuda(),target.cuda()
        # data (batch_size, step, 88)
        # target (batch_size, 1)
        data_1, data_2,data_3, target = Variable(data_1), Variable(data_2),Variable(data_3),Variable(target)
        target = target.squeeze()
        #dia_out_a, _ = dia_net_a(data_2)
        #dia_out_b, _ = dia_net_b(data_3)
        dia_out_all, _ = dia_net_all(data_1)
        #line_input_1 = torch.cat((dia_out_a,dia_out_b), 1)
        #line_input = torch.cat((line_input_1,dia_out_all), 1)
        #line_out = output_net(dia_out_all)

        #dia_net_a_optimizer.zero_grad()
        #dia_net_b_optimizer.zero_grad()
        dia_net_all_optimizer.zero_grad()
        #output_net_optimizer.zero_grad()

        loss = torch.nn.CrossEntropyLoss()(dia_out_all, target.long())

        loss.backward()

        #dia_net_a_optimizer.step()
        #dia_net_b_optimizer.step()
        dia_net_all_optimizer.step()
        #output_net_optimizer.step()

        train_loss += loss

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval
            ))
            train_loss = 0


def Test(test_len):
    #dia_net_a.eval()
    #dia_net_b.eval()
    dia_net_all.eval()
    #output_net.eval()

    label_pre = []
    label_true = []
    with torch.no_grad():
        for batch_idx, (data_1,data_2,data_3,target) in enumerate(test_loader):
            if args.cuda:
                data_1, data_2, data_3, target = data_1.cuda(), data_2.cuda(), data_3.cuda(), target.cuda()
            data_1, data_2, data_3, target = Variable(data_1), Variable(data_2), Variable(data_3), Variable(target)
            #dia_out_a, _ = dia_net_a(data_2)
            #dia_out_b, _ = dia_net_b(data_3)
            dia_out_all, _ = dia_net_all(data_1)
            #line_input_1 = torch.cat((dia_out_a, dia_out_b), 1)
            #line_input = torch.cat((line_input_1, dia_out_all), 1)
            #line_out = output_net(dia_out_all)
            output = torch.argmax(dia_out_all, dim=1)
            label_true.extend(target.cpu().data.numpy())
            label_pre.extend(output.cpu().data.numpy())
        accuracy_recall = recall_score(label_true[:test_len], label_pre[:test_len], average='macro')
        accuracy_f1 = metrics.f1_score(label_true[:test_len], label_pre[:test_len], average='macro')
        CM_test = confusion_matrix(label_true[:test_len], label_pre[:test_len])
        print(accuracy_recall)
        print(accuracy_f1)
        print(CM_test)
    return accuracy_f1, accuracy_recall, label_pre, label_true


Final_result = []
Fineal_f1 = []
kf = KFold(n_splits=5)
for index, (train, test) in enumerate(kf.split(data)):
    print(index)

    train_loader, test_loader, input_test_data_id, input_test_label_org, test_len = Get_data(data, train, test, args)

    #dia_net_a = Utterance_net(args.utt_insize, args)
    #dia_net_b = Utterance_net(args.utt_insize, args)
    dia_net_all = Dialogue_net(args.utt_insize, args)
    #output_net = Output_net(512, args)

    if args.cuda:
        #dia_net_a = dia_net_a.cuda()
        #dia_net_b = dia_net_b.cuda()
        dia_net_all = dia_net_all.cuda()
        #output_net = output_net.cuda()

    lr = args.lr
    #dia_net_a_optimizer = getattr(optim, args.optim)(dia_net_a.parameters(), lr=lr)
    #dia_net_b_optimizer = getattr(optim, args.optim)(dia_net_b.parameters(), lr=lr)
    dia_net_all_optimizer = getattr(optim, args.optim)(dia_net_all.parameters(), lr=lr)
    #output_net_optimizer = getattr(optim, args.optim)(output_net.parameters(), lr=lr)

    #dia_net_a_optim = optim.Adam(dia_net_a.parameters(), lr=lr)
    #dia_net_b_optim = optim.Adam(dia_net_b.parameters(), lr=lr)
    dia_net_all_optim = optim.Adam(dia_net_all.parameters(), lr=lr)
    #output_net_optim = optim.Adam(output_net.parameters(), lr=lr)

    f1 = 0
    recall = 0
    for epoch in range(1, args.epochs + 1):
        Train(epoch)
        accuracy_f1, accuracy_recall, pre_label, true_label = Test(test_len)
        if epoch % 15 == 0:
            lr /= 10
            '''
            for param_group in dia_net_a_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in dia_net_b_optimizer.param_groups:
                param_group['lr'] = lr
            '''
            for param_group in dia_net_all_optimizer.param_groups:
                param_group['lr'] = lr
            '''
            for param_group in output_net_optimizer.param_groups:
                param_group['lr'] = lr
            '''

        if (accuracy_f1 > f1 and accuracy_recall > recall):
            predict = copy.deepcopy(input_test_label_org)
            num = 0
            for x in range(len(predict)):
                predict[x] = pre_label[num]
                num = num + 1
            result_label = predict
            recall = accuracy_recall
    onegroup_result = []
    for i in range(len(input_test_data_id)):
        a = {}
        a['id'] = input_test_data_id[i]
        a['Predict_label'] = pre_label[i]
        a['True_label'] = input_test_label_org[i]
        onegroup_result.append(a)
    Final_result.append(onegroup_result)
    Fineal_f1.append(accuracy_f1)

file = open('Final_result.pickle', 'wb')
pickle.dump(Final_result, file)
file.close()
file = open('Fineal_f1.pickle', 'wb')
pickle.dump(Fineal_f1, file)
file.close()