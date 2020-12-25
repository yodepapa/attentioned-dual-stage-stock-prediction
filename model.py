import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class AttnEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, time_step):
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = time_step
        #input_size一般指词向量的维度。
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        
        """获得特征维度的权重"""
        #linear层只是乘矩阵W，并不激活。输入为h t-1拼接s t-1向量，输出是时间序列的长度
        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=self.T)
        #输入是某一特征的一个时间序列，输出是一个时间序列是长度
        self.attn2 = nn.Linear(in_features=self.T, out_features=self.T)
        #attn1和attn2求和后激活一下
        self.tanh = nn.Tanh()
        #激活完再乘一个矩阵，获得想要的维度，也就是某个特征的全权重值。
        self.attn3 = nn.Linear(in_features=self.T, out_features=1)
        #self.attn = nn.Sequential(attn1, attn2, nn.Tanh(), attn3)


    def forward(self, driving_x):
        #batch_size相当于一个小数据集的大小，相当于数据量。
        batch_size = driving_x.size(0)
        # batch_size * time_step * hidden_size
        code = self.init_variable(batch_size, self.T, self.hidden_size)
        # initialize hidden state
        h = self.init_variable(1, batch_size, self.hidden_size)
        # initialize cell state
        s = self.init_variable(1, batch_size, self.hidden_size)
        for t in range(self.T):
            # batch_size * input_size * (2 * hidden_size + time_step)
            x = torch.cat((self.embedding_hidden(h), self.embedding_hidden(s)), 2)
            #x是三维的数组，为什么Linear（）可以接收？？？？
            z1 = self.attn1(x)
            #Linear接收第一维度默认为batch
            #driving_x 应该是没个横行是一个时刻，每一列是一个特征。所以需要转置为每一个横行是一个特征。
            z2 = self.attn2(driving_x.permute(0, 2, 1))
            x = z1 + z2
            # batch_size * input_size * 1
            z3 = self.attn3(self.tanh(x))
            if batch_size > 1:
                attn_w = F.softmax(z3.view(batch_size, self.input_size), dim=1)
            else:
                attn_w = self.init_variable(batch_size, self.input_size) + 1
            # batch_size * input_size 给各个时刻，各个batch，各个特征的注意力权重
            weighted_x = torch.mul(attn_w, driving_x[:, t, :])
            _, states = self.lstm(weighted_x.unsqueeze(0), (h, s))
            h = states[0]
            s = states[1]

            # encoding result
            # batch_size * time_step * encoder_hidden_size
            code[:, t, :] = h

        return code

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.repeat(self.input_size, 1, 1).permute(1, 0, 2)


class AttnDecoder(nn.Module):

    def __init__(self, code_hidden_size, hidden_size, time_step):
        super(AttnDecoder, self).__init__()
        self.code_hidden_size = code_hidden_size
        self.hidden_size = hidden_size
        self.T = time_step

        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size)
        self.attn2 = nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=code_hidden_size, out_features=1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size)
        self.tilde = nn.Linear(in_features=self.code_hidden_size + 1, out_features=1)
        self.fc1 = nn.Linear(in_features=code_hidden_size + hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, h, y_seq):
        batch_size = h.size(0)
        d = self.init_variable(1, batch_size, self.hidden_size)
        s = self.init_variable(1, batch_size, self.hidden_size)
        ct = self.init_variable(batch_size, self.hidden_size)

        for t in range(self.T):
            # batch_size * time_step * (encoder_hidden_size + decoder_hidden_size)
            x = torch.cat((self.embedding_hidden(d), self.embedding_hidden(s)), 2)
            z1 = self.attn1(x)
            z2 = self.attn2(h)
            x = z1 + z2
            # batch_size * time_step * 1
            z3 = self.attn3(self.tanh(x))
            if batch_size > 1:
                beta_t = F.softmax(z3.view(batch_size, -1), dim=1)
            else:
                beta_t = self.init_variable(batch_size, self.code_hidden_size) + 1
            # batch_size * encoder_hidden_size
            #.bmm()是矩阵乘,时间维度得权重
            ct = torch.bmm(beta_t.unsqueeze(1), h).squeeze(1)
            if t < self.T - 1:
                yc = torch.cat((y_seq[:, t].unsqueeze(1), ct), dim=1)
                y_tilde = self.tilde(yc)
                _, states = self.lstm(y_tilde.unsqueeze(0), (d, s))
                d = states[0]
                s = states[1]
        # batch_size * 1
        y_res = self.fc2(self.fc1(torch.cat((d.squeeze(0), ct), dim=1)))
        return y_res

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.repeat(self.T, 1, 1).permute(1, 0, 2)


