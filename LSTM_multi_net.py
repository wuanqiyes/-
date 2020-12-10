import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import get_data

### LSTM 实现时间序列预测

# 预设变量
xl_index = 3    # 读取xl文件的序列

continuous_seq_10_length = 10 #连续周期为10的序列长度
continuous_seq_5_length = 5 #连续周期为5的序列长度
continuous_seq_3_length = 3 #连续周期为3的序列长度
gap_1_seq_5_length = 3 #间隔为1周期为5的序列长度
gap_1_seq_9_length = 5 #间隔为1周期为9的序列长度
gap_2_seq_7_length = 3 #间隔为2周期为7的序列长度
gap_3_seq_9_length = 3 #间隔为3周期为9的序列长度
gap_6_seq_21_length = 3 #间隔一周的序列长度

def SeriesGen(N):
    x = torch.arange(1,N,0.01)
    print(len(x))
    return torch.sin(x)

# sep:原始数据, k:周期序列长度, gap:周期序列间隔
def trainDataGen_gap0(seq,k,gap):
    dat = list()
    L = len(seq)
    for i in range(L-k-gap):
        indat = list()
        outdat = list()
        count = gap+1
        for j in range(k):
            if count % (gap+1) == 0:
                indat.append(seq[i+j])
                outdat.append(seq[i+j+1+gap])
            count+= 1
        dat.append((indat,outdat))
    return dat

def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)

class LSTMpred(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(LSTMpred, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)
        outdat = self.hidden2out(lstm_out.view(len(seq), -1))
        return outdat


# 读取数据
case_count  = get_data.get_xldata(xl_index)
# 归一化处理
max_v = max(case_count)
new_case_count = [x/(max_v) for x in case_count]
# 生成序列文件
y = torch.tensor(new_case_count)
dat_10_0 = trainDataGen_gap0(y.numpy(),10,0)
dat_5_0 = trainDataGen_gap0(y.numpy(),5,0)
dat_3_0 = trainDataGen_gap0(y.numpy(),3,0)
dat_5_1 = trainDataGen_gap0(y.numpy(),5,1)
dat_9_1 = trainDataGen_gap0(y.numpy(),9,1)
dat_7_2 = trainDataGen_gap0(y.numpy(),7,2)
dat_9_3 = trainDataGen_gap0(y.numpy(),9,3)

# 模型训练参数设置
model_10_0 = LSTMpred(1,6) # 输入层和隐藏层尺寸
model_5_0 = LSTMpred(1,6) # 输入层和隐藏层尺寸
model_3_0 = LSTMpred(1,6) # 输入层和隐藏层尺寸
model_5_1 = LSTMpred(1,6) # 输入层和隐藏层尺寸
model_9_1 = LSTMpred(1,6) # 输入层和隐藏层尺寸
model_7_2 = LSTMpred(1,6) # 输入层和隐藏层尺寸
model_9_3 = LSTMpred(1,6) # 输入层和隐藏层尺寸

loss_function = nn.MSELoss() # 损失函数
optimizer_10_0 = optim.SGD(model_10_0.parameters(), lr=0.01) # 优化器设置，SGD方法，学习率0.01
optimizer_5_0 = optim.SGD(model_5_0.parameters(), lr=0.01) # 优化器设置，SGD方法，学习率0.01
optimizer_3_0 = optim.SGD(model_3_0.parameters(), lr=0.01) # 优化器设置，SGD方法，学习率0.01
optimizer_5_1 = optim.SGD(model_5_1.parameters(), lr=0.01) # 优化器设置，SGD方法，学习率0.01
optimizer_9_1 = optim.SGD(model_9_1.parameters(), lr=0.01) # 优化器设置，SGD方法，学习率0.01
optimizer_7_2 = optim.SGD(model_7_2.parameters(), lr=0.01) # 优化器设置，SGD方法，学习率0.01
optimizer_9_3 = optim.SGD(model_9_3.parameters(), lr=0.01) # 优化器设置，SGD方法，学习率0.01

num_epochs = 500# 迭代次数

for epoch in range(num_epochs):
    print("迭代训练：",epoch)
    loss_sum = 0.0
    for seq, outs in dat_10_0[:]:
        seq = ToVariable(seq) # 转 tensor
        outs = ToVariable(outs)  # 转 tensor
        optimizer_10_0.zero_grad() # 清空梯度
        model_10_0.hidden = model_10_0.init_hidden()
        modout = model_10_0(seq) # 模型计算
        loss = loss_function(modout, outs) # 计算loss
        loss_sum += loss
        loss.backward()
        optimizer_10_0.step()
    # if epoch % 20 == 0 and epoch > 0:
    #     loss = loss_sum / len(dat)
    #     print("迭代训练",epoch,":",loss)

    for seq, outs in dat_5_0[:]:
        seq = ToVariable(seq) # 转 tensor
        outs = ToVariable(outs)  # 转 tensor
        optimizer_5_0.zero_grad() # 清空梯度
        model_5_0.hidden = model_5_0.init_hidden()
        modout = model_5_0(seq) # 模型计算
        loss = loss_function(modout, outs) # 计算loss
        loss_sum += loss
        loss.backward()
        optimizer_5_0.step()

    for seq, outs in dat_3_0[:]:
        seq = ToVariable(seq) # 转 tensor
        outs = ToVariable(outs)  # 转 tensor
        optimizer_3_0.zero_grad() # 清空梯度
        model_3_0.hidden = model_3_0.init_hidden()
        modout = model_3_0(seq) # 模型计算
        loss = loss_function(modout, outs) # 计算loss
        loss_sum += loss
        loss.backward()
        optimizer_3_0.step()

    for seq, outs in dat_5_1[:]:
        seq = ToVariable(seq) # 转 tensor
        outs = ToVariable(outs)  # 转 tensor
        optimizer_5_1.zero_grad() # 清空梯度
        model_5_1.hidden = model_5_1.init_hidden()
        modout = model_5_1(seq) # 模型计算
        loss = loss_function(modout, outs) # 计算loss
        loss_sum += loss
        loss.backward()
        optimizer_5_1.step()

    for seq, outs in dat_9_1[:]:
        seq = ToVariable(seq) # 转 tensor
        outs = ToVariable(outs)  # 转 tensor
        optimizer_9_1.zero_grad() # 清空梯度
        model_9_1.hidden = model_9_1.init_hidden()
        modout = model_9_1(seq) # 模型计算
        loss = loss_function(modout, outs) # 计算loss
        loss_sum += loss
        loss.backward()
        optimizer_9_1.step()

    for seq, outs in dat_7_2[:]:
        seq = ToVariable(seq) # 转 tensor
        outs = ToVariable(outs)  # 转 tensor
        optimizer_7_2.zero_grad() # 清空梯度
        model_7_2.hidden = model_7_2.init_hidden()
        modout = model_7_2(seq) # 模型计算
        loss = loss_function(modout, outs) # 计算loss
        loss_sum += loss
        loss.backward()
        optimizer_7_2.step()

    for seq, outs in dat_9_3[:]:
        seq = ToVariable(seq) # 转 tensor
        outs = ToVariable(outs)  # 转 tensor
        optimizer_9_3.zero_grad() # 清空梯度
        model_9_3.hidden = model_9_3.init_hidden()
        modout = model_9_3(seq) # 模型计算
        loss = loss_function(modout, outs) # 计算loss
        loss_sum += loss
        loss.backward()
        optimizer_9_3.step()





# 预测现有数据
predDat_10_0 = list()
for seq, trueVal in dat_10_0[:]:
    #print("seq=",seq)
    #print("trueval=",trueVal)
    seq = ToVariable(seq)
    trueVal = ToVariable(trueVal)
    modout = model_10_0(seq)
    pre = modout[-1].data.numpy()[0] #从 index = 5 开始
    #print(pre)
    loss = loss_function(modout, outs)
    loss_sum += loss
    predDat_10_0.append(model_10_0(seq)[-1].data.numpy()[0])

predDat_5_0 = list()
for seq, trueVal in dat_5_0[:]:
    seq = ToVariable(seq)
    trueVal = ToVariable(trueVal)
    modout = model_5_0(seq)
    pre = modout[-1].data.numpy()[0] #从 index = 5 开始
    loss = loss_function(modout, outs)
    loss_sum += loss
    predDat_5_0.append(model_5_0(seq)[-1].data.numpy()[0])

predDat_3_0 = list()
for seq, trueVal in dat_3_0[:]:
    seq = ToVariable(seq)
    trueVal = ToVariable(trueVal)
    modout = model_3_0(seq)
    pre = modout[-1].data.numpy()[0] #从 index = 5 开始
    loss = loss_function(modout, outs)
    loss_sum += loss
    predDat_3_0.append(model_3_0(seq)[-1].data.numpy()[0])

predDat_5_1 = list()
for seq, trueVal in dat_5_1[:]:
    seq = ToVariable(seq)
    trueVal = ToVariable(trueVal)
    modout = model_5_1(seq)
    pre = modout[-1].data.numpy()[0] #从 index = 5 开始
    loss = loss_function(modout, outs)
    loss_sum += loss
    predDat_5_1.append(model_5_1(seq)[-1].data.numpy()[0])

predDat_9_1 = list()
for seq, trueVal in dat_9_1[:]:
    seq = ToVariable(seq)
    trueVal = ToVariable(trueVal)
    modout = model_9_1(seq)
    pre = modout[-1].data.numpy()[0] #从 index = 5 开始
    loss = loss_function(modout, outs)
    loss_sum += loss
    predDat_9_1.append(model_9_1(seq)[-1].data.numpy()[0])

predDat_7_2 = list()
for seq, trueVal in dat_7_2[:]:
    seq = ToVariable(seq)
    trueVal = ToVariable(trueVal)
    modout = model_7_2(seq)
    pre = modout[-1].data.numpy()[0] #从 index = 5 开始
    loss = loss_function(modout, outs)
    loss_sum += loss
    predDat_7_2.append(model_7_2(seq)[-1].data.numpy()[0])

predDat_9_3 = list()
for seq, trueVal in dat_9_3[:]:
    seq = ToVariable(seq)
    trueVal = ToVariable(trueVal)
    modout = model_9_3(seq)
    pre = modout[-1].data.numpy()[0] #从 index = 5 开始
    loss = loss_function(modout, outs)
    loss_sum += loss
    predDat_9_3.append(model_9_3(seq)[-1].data.numpy()[0])


print("10_0:",len(dat_10_0),"; 5_0:",len(dat_5_0),"; 3_0:",len(dat_3_0),
      "; 5_1:",len(dat_5_1),"; 9_1:",len(dat_9_1),"; 7_2:",len(dat_7_2),"; 9_3:",len(dat_9_3))


#test
# for val in y.numpy()[12:12+3]:
#     print(val)
# print("------")
# for _,val in dat_10_0[12-10:12-10+3]:
#     print(val[-1])
# print("------")
# for _,val in dat_5_0[12-5:12-5+3]:
#     print(val[-1])
# print("------")
# for _,val in dat_3_0[12-3:12-3+3]:
#     print(val[-1])
# print("------")
# for _,val in dat_5_1[12-6:12-6+3]:
#     print(val[-1])
# print("------")
# for _,val in dat_9_1[12-10:12-10+3]:
#     print(val[-1])
# print("------")
# for _,val in dat_7_2[12-9:12-9+3]:
#     print(val[-1])
# print("------")
# for _,val in dat_9_3[12-12:12-12+3]:
#     print(val[-1])

# 制作测试标签
t_labels = y.numpy()[12:]
t_dat_10_0, t_dat_5_0 , t_dat_3_0, t_dat_5_1, t_dat_9_1, t_dat_7_2, t_dat_9_3 \
    = list(),list(),list(),list(),list(),list(),list()
#加载各个模型的预测值
for val,_ in dat_10_0[12-10:]:
    val = ToVariable(val)
    modout = model_10_0(val)
    pre = modout[-1].data.numpy()[0]
    t_dat_10_0.append(pre)
for val,_ in dat_5_0[12-5:]:
    val = ToVariable(val)
    modout = model_5_0(val)
    pre = modout[-1].data.numpy()[0]
    t_dat_5_0.append(pre)
for val,_ in dat_3_0[12-3:]:
    val = ToVariable(val)
    modout = model_3_0(val)
    pre = modout[-1].data.numpy()[0]
    t_dat_3_0.append(pre)
for val,_ in dat_5_1[12-6:]:
    val = ToVariable(val)
    modout = model_5_1(val)
    pre = modout[-1].data.numpy()[0]
    t_dat_5_1.append(pre)
for val, _ in dat_9_1[12 - 10:]:
    val = ToVariable(val)
    modout = model_9_1(val)
    pre = modout[-1].data.numpy()[0]
    t_dat_9_1.append(pre)
for val, _ in dat_7_2[12 - 9:]:
    val = ToVariable(val)
    modout = model_7_2(val)
    pre = modout[-1].data.numpy()[0]
    t_dat_7_2.append(pre)
for val, _ in dat_9_3[12 - 12:]:
    val = ToVariable(val)
    modout = model_9_3(val)
    pre = modout[-1].data.numpy()[0]
    t_dat_9_3.append(pre)

wl_10_0,wl_5_0,wl_3_0,wl_5_1,wl_9_1,wl_7_2,wl_9_3 = list(),list(),list(),list(),list(),list(),list()
X_list=list()
for i in range(len(t_dat_9_3)):
    # 求平均值
    # tmp = 0.0
    # tmp += t_dat_10_0[i]
    # tmp += t_dat_5_0[i]
    # tmp += t_dat_3_0[i]
    # tmp += t_dat_5_1[i]
    # tmp += t_dat_9_1[i]
    # tmp += t_dat_7_2[i]
    # tmp += t_dat_9_3[i]
    # X_list.append(tmp/7)
    # 根据误差获取权重
    loss_10_0 = 1/abs(t_dat_10_0[i] - t_labels[i])
    loss_5_0 = 1 / abs(t_dat_5_0[i] - t_labels[i])
    loss_3_0 = 1 / abs(t_dat_3_0[i] - t_labels[i])
    loss_5_1 = 1 / abs(t_dat_5_1[i] - t_labels[i])
    loss_9_1 = 1 / abs(t_dat_9_1[i] - t_labels[i])
    loss_7_2 = 1 / abs(t_dat_7_2[i] - t_labels[i])
    loss_9_3 = 1 / abs(t_dat_9_3[i] - t_labels[i])
    loss_sum = loss_10_0+loss_5_0+loss_3_0+loss_5_1+loss_9_1+loss_7_2+loss_9_3
    wl_10_0.append(loss_10_0/loss_sum)
    wl_5_0.append(loss_5_0/loss_sum)
    wl_3_0.append(loss_3_0/loss_sum)
    wl_5_1.append(loss_5_1/loss_sum)
    wl_9_1.append(loss_9_1/loss_sum)
    wl_7_2.append(loss_7_2/loss_sum)
    wl_9_3.append(loss_9_3/loss_sum)

# 求权重
w_10_0 = np.mean(wl_10_0)
w_5_0 = np.mean(wl_5_0)
w_3_0 = np.mean(wl_3_0)
w_5_1 = np.mean(wl_5_1)
w_9_1 = np.mean(wl_9_1)
w_7_2 = np.mean(wl_7_2)
w_9_3 = np.mean(wl_9_3)
w_sum = w_10_0+w_5_0+w_3_0+w_5_1+w_9_1+w_7_2+w_9_3
w_10_0 = w_10_0 / w_sum
w_5_0 = w_5_0 / w_sum
w_3_0 = w_3_0 / w_sum
w_5_1 = w_5_1 / w_sum
w_9_1 = w_9_1 / w_sum
w_7_2 = w_7_2 / w_sum
w_9_3 = w_9_3 / w_sum

# 预测现在值
for i in range(len(t_dat_9_3)):
    res = w_10_0 * t_dat_10_0[i] + w_5_0 * t_dat_5_0[i] + w_3_0 * t_dat_3_0[i] + w_5_1 * t_dat_5_1[i] + w_9_1 * t_dat_9_1[i] + w_7_2 * t_dat_7_2[i] + w_9_3 * t_dat_9_3[i]
    X_list.append(res)

#预测未来数据 --采用连续模型
#预测未来数据
pre_day_count = 8
seq,trueVal = dat_10_0[-1]
pred_10_0 = []
for i in range(pre_day_count):
    seq_val = ToVariable(seq)
    pre = model_10_0(seq_val)[-1].data.numpy()[0]
    pred_10_0.append(pre)
    seq=np.delete(seq, 0)
    seq=np.append(seq,pre)

seq,trueVal = dat_5_0[-1]
pred_5_0 = []
for i in range(pre_day_count):
    seq_val = ToVariable(seq)
    pre = model_5_0(seq_val)[-1].data.numpy()[0]
    pred_5_0.append(pre)
    seq=np.delete(seq, 0)
    seq=np.append(seq,pre)

seq,trueVal = dat_3_0[-1]
pred_3_0 = []
for i in range(pre_day_count):
    seq_val = ToVariable(seq)
    pre = model_3_0(seq_val)[-1].data.numpy()[0]
    pred_3_0.append(pre)
    seq=np.delete(seq, 0)
    seq=np.append(seq,pre)

# 开始间隔一的预测
seq_1,trueVal_1 = dat_5_1[-1]
seq,trueVal = dat_5_1[-2]
pred_5_1 = []
for i in range(pre_day_count):
    seq_val = ToVariable(seq)
    pre = model_5_1(seq_val)[-1].data.numpy()[0]
    pred_5_1.append(pre)
    seq = np.delete(seq, 0)
    seq = np.append(seq, pre)
    tmp1,tmp2 = seq_1 ,trueVal_1
    seq_1,trueVal_1 = seq,trueVal
    seq, trueVal = tmp1,tmp2

seq_1,trueVal_1 = dat_9_1[-1]
seq,trueVal = dat_9_1[-2]
pred_9_1 = []
for i in range(pre_day_count):
    seq_val = ToVariable(seq)
    pre = model_9_1(seq_val)[-1].data.numpy()[0]
    pred_9_1.append(pre)
    seq = np.delete(seq, 0)
    seq = np.append(seq, pre)
    tmp1,tmp2 = seq_1 ,trueVal_1
    seq_1,trueVal_1 = seq,trueVal
    seq, trueVal = tmp1,tmp2

# 开始间隔2的预测
seq_2,trueVal_2 = dat_7_2[-1]
seq_1,trueVal_1 = dat_7_2[-2]
seq,trueVal = dat_7_2[-3]
pred_7_2 = []
for i in range(pre_day_count):
    seq_val = ToVariable(seq)
    pre = model_7_2(seq_val)[-1].data.numpy()[0]
    pred_7_2.append(pre)
    seq = np.delete(seq, 0)
    seq = np.append(seq, pre)
    tmp1,tmp2 = seq_2,trueVal_2
    seq_2,trueVal_2 = seq_1,trueVal_1
    seq_1, trueVal_1 = seq, trueVal
    seq, trueVal = tmp1,tmp2

# 开始间隔3的预测
seq_3,trueVal_3 = dat_9_3[-1]
seq_2,trueVal_2 = dat_9_3[-2]
seq_1,trueVal_1 = dat_9_3[-3]
seq,trueVal = dat_9_3[-4]
pred_9_3 = []
for i in range(pre_day_count):
    seq_val = ToVariable(seq)
    pre = model_9_3(seq_val)[-1].data.numpy()[0]
    pred_9_3.append(pre)
    seq = np.delete(seq, 0)
    seq = np.append(seq, pre)
    tmp1,tmp2 = seq_3,trueVal_3
    seq_3, trueVal_3 = seq_2,trueVal_2
    seq_2,trueVal_2 = seq_1,trueVal_1
    seq_1, trueVal_1 = seq, trueVal
    seq, trueVal = tmp1,tmp2

final_pre = []
for i in range(pre_day_count):

    # 去极端值求均值
    # tmp = 0.0
    # tmp += pred_10_0[i]
    # tmp += pred_5_0[i]
    # tmp += pred_3_0[i]
    # tmp += pred_5_1[i]
    # tmp += pred_9_1[i]
    # tmp += pred_7_2[i]
    # tmp += pred_9_3[i]
    #
    # tmp -= max(pred_10_0[i],pred_5_0[i],pred_3_0[i],pred_5_1[i],pred_9_1[i],pred_7_2[i],pred_9_3[i])
    # tmp -= min(pred_10_0[i],pred_5_0[i],pred_3_0[i],pred_5_1[i],pred_9_1[i],pred_7_2[i],pred_9_3[i])
    # final_pre.append(tmp / 5)
    # X_list.append(tmp / 5)
    # print("pre:", tmp * max_v / 5)
    #比较误差争取权重法
    res = w_10_0*pred_10_0[i] + w_5_0*pred_5_0[i] + w_3_0*pred_3_0[i] + w_5_1*pred_5_1[i]+ w_9_1*pred_9_1[i] + w_7_2*pred_7_2[i] + w_9_3*pred_9_3[i]


    final_pre.append(res)
    X_list.append(res)
    print("pre:",res * max_v)


# fig = plt.figure()

true_y = []
true_pre= []
for val in y.tolist():
    val *= max_v
    true_y.append(val)
for val in X_list:
    val *= max_v
    true_pre.append(val)
# print(true_pre)
# 图像显示
plt.plot(true_y)
plt.plot(range(len(true_y)-len(true_pre)+pre_day_count-1, len(true_y)+pre_day_count-1), true_pre)
plt.show()



