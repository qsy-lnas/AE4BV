import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
import re
import os

class AE4BV(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer, dropout = 0.2):
        super(AE4BV, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.W = nn.Parameter((torch.randn(output_dim, input_dim) * 1e-2).to(self.device))
        self.a = nn.Parameter(torch.zeros(input_dim).to(self.device))
        self.int_dim = input_dim #input dimension就是数据的dimension，即integer的个数
        self.h_dim = output_dim #中间的特征向量（即encoder的output）的维度
        self.hidden_dim = hidden_layer
        
        #Encoder的结构
        layers = []
        in_dim = input_dim
        for dim in self.hidden_dim:
            layers.append(nn.Sequential(nn.Linear(in_dim, dim), nn.LeakyReLU())) #基本结构是全连接层+激活函数
            in_dim = dim
        self.encoder_layer = nn.ModuleList(layers).to(self.device)
        self.dropout = dropout
    
    def encoder(self, temp):
        res_en = temp #temp是操作到当前的结果，res_en是encoder处理后的result
        for i, module in enumerate(self.encoder_layer):
            #print(i)
            if(self.hidden_dim[i] == self.int_dim): #类似于Resnet的跳跃连接
                if i != 0: #最开始不进行此操作
                    res_en = module(res_en) + temp
                    temp = res_en
            else:
                res_en = module(res_en)
        return res_en
    
    def decoder(self, h): #h是encoder输出的特征向量；decoder是一个linear层+sigmoid激活函数
        res_de = nn.functional.linear(h, self.W.t(), self.a)
        return nn.functional.sigmoid(res_de)
    
    def forward(self, ip):
        res_en = self.encoder(ip)
        res_de = self.decoder(res_en)
        return res_en, res_de

def generator(path, num):
    filename = os.listdir(path)[:num]
    
    val = []
    for file in filename:
        if not re.match('.*\.sol$', file):
            continue
        int_val = []
        with open(path + '/' + file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('#'):
                    continue #第一行的objective前面有井号
                line = re.sub('\n$', '', line) #把每行最后的空格substitute成没有（即删掉）
                name = line[:line.find('[')] #提取变量名，即直到[以前的
                if not re.match('Make|Utilization', name):
                    continue #只保留整数变量
                int_val.append(round(float(line.split(' ')[1]))) #取空格后的，即0-1变量
        val.append(int_val)
    return val

def plot_wrong(wrong_pct):
    plt.plot(range(len(wrong_pct)), wrong_pct, label = '验证集上错误概率')
    plt.legend()
    # plt.title('验证集上错误概率')
    # plt.xlabel('训练轮次 / 轮')
    # plt.ylabel('错误概率')
    # plt.savefig('Model/Wrong_percentage_913.jpg')
    plt.show()

def plot_train(train_loss):
    plt.plot(range(len(train_loss)), train_loss, label = '训练过程损失曲线')
    plt.legend()
    # plt.title('训练过程损失曲线')
    # plt.xlabel('训练轮次 / 轮')
    # plt.ylabel('损失')
    # plt.savefig('Model/Train_loss_913.jpg')
    plt.show()

def train(model, epochs, batch_size, alpha, savename, data_range):
        #加载训练数据
        train_path = f"data/Data_Train_{data_range}"
        test_path = f"data/Data_Test_{data_range}"
        data_train = generator(train_path, 1000)
        data_test = generator(test_path, 200)
        data_train = torch.tensor(data_train).float().to(model.device)
        data_test = torch.tensor(data_test).float().to(model.device)
        #print(data.size())
        
        # l_train = int(len(data)*0.8) #划分训练集和验证集
        # l_val=len(data)-l_train
        # train_data, val_data = torch.utils.data.random_split(data,[l_train,l_val])
        train_loader = torch.utils.data.DataLoader(data_train, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(data_test, batch_size = 200, shuffle = False)
        
        loss = nn.BCELoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr = alpha)
        
        min_loss = 1e5 #用来记录最小的误差
        W = []
        a = []
        h = []
        wrong = []
        wrong_record = [] #验证集预测错误率的record
        loss_epoch = [] #总体的loss
        sa_record = []
        rec = np.zeros(216) #各个维度的错误率的record
        
        for epoch in range(epochs):
            loss_record = []
            model.train()
            for t, data in enumerate(train_loader):
                #print(data.size())
                i, y_pred = model(data)
                l = loss(y_pred, data)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                loss_record.append(l.data.cpu())
            #print(epoch)
            h, W, a, wrong_sum, wrong, min_loss, rec, sa = evaluation(model, val_loader, loss_record, min_loss, W, a, h, epoch, wrong, rec)
            loss_epoch.append(np.mean(loss_record))
            wrong_record.append(wrong)
            sa_record.append(sa) 
        #保存
        # torch.save(torch.le(wrong, 100), 'Model/{}to{}_int_dim_925.pt'.format(model.int_dim, model.h_dim))
        torch.save(W, 'Model/{}to{}_W_{}_{}.pt'.format(model.int_dim, model.h_dim, data_range, savename))
        torch.save(a, 'Model/{}to{}_a_{}_{}.pt'.format(model.int_dim, model.h_dim, data_range, savename))
        # torch.save(h, 'Model/{}to{}_h_925.pt'.format(model.int_dim, model.h_dim))
        
        #画图
        # plot_train(loss_epoch)
        # plot_wrong(wrong_record)
        
        W = np.array(W.cpu().detach())
        a = np.array(a.cpu().detach()).reshape([-1, 1])
        h = np.array(h.cpu().detach())
        
        print("Best HL:  ", np.min(np.array(wrong_record)))
        print("Best PPO: ", cal_PPO(data_range, savename, W, a, test_path))
        # print(np.max(sa_record))
        
        pred = W.T @ h.T + a
        print('Pred Max = ', pred.max(), 'Pred Min = ', pred.min())

def evaluation(model, loader, loss_record, min_loss, W, a, h, epoch, wrong, rec):
        model.eval()
        loss = nn.BCELoss()
        wrong_sum = 0
        
        for t, data in enumerate(loader):
            _, y_pred = model(data)
            l = loss(y_pred, data).data.cpu()
            rec = np.zeros(data.cpu().shape[1])
            if l < min_loss:
                min_loss = l
                h = model.encoder(data) #得到当前情况对应的最好参数h、W、a
                W = model.W
                a = model.a
            
            wrong_sum = torch.sum(~torch.eq(torch.ge(y_pred, 0.5), data).cpu()) # 错误维度数
            wrong = wrong_sum / (data.cpu().shape[0] * data.cpu().shape[1])#记录各个维度的错误率
            # for i, (x, y) in enumerate(zip(torch.ge(y_pred, 0.5), data)):
            rec = rec + np.sum(np.array(~torch.eq(torch.ge(y_pred, 0.5), data).cpu()), axis = 0)
            sa = (200 - np.sum(np.greater_equal( np.sum(np.array(~torch.eq(torch.ge(y_pred, 0.5), data).cpu()), axis = 1), 0.5)) )/ 200
                
            if (epoch + 1) % 50 == 0 and t == 0:
                print('Epoch = %d, Training Loss = %.4f, Evaluating Loss = %.4f, Best Evaluating Loss = %.4f, HL = %.4f' % (epoch + 1, np.mean(loss_record), l, min_loss, wrong))   
        return h, W, a, wrong_sum, wrong , min_loss, rec, sa

def test_in_polytope(param, M_l, M_h, sol):
    W = torch.load(f'model/216to20_W_{param}.pt')
    a = torch.load(f'model/216to20_a_{param}.pt')
    W = np.array(W.cpu().detach())
    a = np.array(a.cpu().detach()).reshape([-1, 1])
    W = W.T

    model = gp.Model()
    model.setParam('OutputFlag', 0)
    #variables
    #the beginning of task i at event point n; binary variable
    # make = model.addVars(tasks, events, vtype = GRB.BINARY, name = 'Make')
    #the utilization of unit j at event point n; binary variable
    # utilization = model.addVars(units, events, vtype = GRB.BINARY, name = 'Utilization')

    h = model.addVars(20, 1, vtype = GRB.CONTINUOUS, name = 'Supplementary_variable')
    for i in range(0, 20):
        h[i, 0].start = np.random.normal(0, 20)

    model.addConstrs((gp.quicksum(W[i, j] * h[j, 0] + a[j] for j in range(0, 20)) >= (1 - sol[i]) * (M_l) for i in range(0, 216)), name = 'Constr0')
    model.addConstrs((gp.quicksum(W[i, j] * h[j, 0] + a[j] for j in range(0, 20)) <= sol[i] * (M_h) for i in range(0, 216)), name = 'Constr5')
    # model.addConstrs((gp.quicksum(W[t * 9 + s, j] * h[j, 0] + a[j] for j in range(0, 10)) >= (1 - make[tasks[t], events[s]]) * (-225) for t in range(0, 16) for s in range(0, 9)), name = 'Constr1')
    # model.addConstrs((gp.quicksum(W[t * 9 + s + 144, j] * h[j, 0] + a[j] for j in range(0, 10)) >= (1 - utilization[units[t], events[s]]) * (-225) for t in range(0, 8) for s in range(0, 9)), name = 'Constr2')
    # model.addConstrs((gp.quicksum(W[t * 9 + s, j] * h[j, 0] + a[j] for j in range(0, 10)) <= make[tasks[t], events[s]] * (210) for t in range(0, 16) for s in range(0, 9)), name = 'Constr3')
    # model.addConstrs((gp.quicksum(W[t * 9 + s + 144, j] * h[j, 0] + a[j] for j in range(0, 10)) <= utilization[units[t], events[s]] * (210) for t in range(0, 8) for s in range(0, 9)), name = 'Constr4')
    model.optimize()
    if model.status == GRB.OPTIMAL:
        return 1
    else: 
        return 0

def cal_PPO(data_range, savename, W, a, testset):
    param = data_range + "_" + savename
    M_l = -GRB.INFINITY
    M_h = GRB.INFINITY
    total_valid = 0
    for i in range(1000, 1200):
        # print(f'data/Data_Test_0{param}/{(i):0>4}.sol')
        test_data = f'{testset}/{(i):0>4}.sol'
        int_val = []
        with open(test_data, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('#'):
                    continue #第一行的objective前面有井号
                line = re.sub('\n$', '', line) #把每行最后的空格substitute成没有（即删掉）
                name = line[:line.find('[')] #提取变量名，即直到[以前的
                if not re.match('Make|Utilization', name):
                    continue #只保留整数变量
                int_val.append(round(float(line.split(' ')[1]))) #取空格后的，即0-1变量
        isvalid = test_in_polytope(param, M_l=M_l, M_h=M_h, sol = int_val)
        total_valid += isvalid
    return (total_valid / 200)

if __name__ == "__main__":
    int_dim = 216
    h_dim = 20
    model_save_name = "default"
    data_range = "5pt"
    hidden_layer = [20, int_dim, 40, int_dim, 120, int_dim, 180, int_dim, h_dim]
    model = AE4BV(input_dim = int_dim, output_dim = h_dim, hidden_layer = hidden_layer, dropout = 0.2)
    train(model, epochs = 1000, batch_size = 4, alpha = 2e-4, savename = model_save_name, data_range= data_range)
