import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# 设备检测
import os
gpu_id = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Xavier 均匀初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        # print("inital  linear weight ")


class word_embedding(nn.Module): # 定义词嵌入模块
    def __init__(self,vocab_length , embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(-1,1,size=(vocab_length ,embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length,embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))
        self.word_embedding.to(device)  # 将嵌入层移动到 GPU

    def forward(self, input_sentence):
        """
        :param input_sentence:  a tensor, containing several word indices.
        :return: a tensor, containing word embeddings.
        """
        input_sentence = input_sentence.to(device)  # 确保输入在 GPU 上
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


class RNN_model(nn.Module):
    def __init__(self, batch_sz ,vocab_len ,word_embedding,embedding_dim, lstm_hidden_dim):
        super(RNN_model,self).__init__()

        self.word_embedding_lookup = word_embedding.to(device)  # 词嵌入模块移动到 GPU
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim

        # 定义 LSTM 层
        self.rnn_lstm = nn.LSTM(input_size=embedding_dim,
                                hidden_size=lstm_hidden_dim,
                                num_layers=2,
                                batch_first=True).to(device)  # 移动到 GPU

        # 全连接层
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len).to(device)

        # 进行权重初始化
        self.apply(weights_init)

        # LogSoftmax 激活函数
        self.softmax = nn.LogSoftmax(dim=1).to(device)


    def forward(self, sentence, is_test=False):
        sentence = sentence.to(device)  # 确保输入在 GPU 上
        batch_input = self.word_embedding_lookup(sentence).view(1, -1, self.word_embedding_dim).to(device)
        # print(batch_input.size()) # print the size of the input:torch.Size([1, 26, 100])
        ################################################
        # here you need to put the "batch_input"  input the self.lstm which is defined before.
        # the hidden output should be named as output, the initial hidden state and cell state set to zero.
        # 初始化隐藏状态和细胞状态为零，层数为2
        h0 = torch.zeros(2, batch_input.size(0), self.lstm_dim, device=device)
        c0 = torch.zeros(2, batch_input.size(0), self.lstm_dim, device=device)
        # 将输入传入 LSTM 层，获得输出和 (hn, cn)
        output, (hn, cn) = self.rnn_lstm(batch_input, (h0, c0))

        ################################################
        # 将 LSTM 的输出调整形状后传入全连接层
        out = output.contiguous().view(-1,self.lstm_dim)

        out =  F.relu(self.fc(out))

        out = self.softmax(out)

        if is_test:
            prediction = out[ -1, : ].view(1,-1)
            output = prediction
        else:
           output = out
        # print(out)
        return output
