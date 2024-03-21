import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd import Variable

# 实现Transformer的整个代码结构。


############################ 嵌入层；#################################
class embeddings(nn.Module):
    def __init__(self,d_model:int,vocab:int):
        # embedding层一共有vocab个词语，每个词语的维度为d_model个，一共是vocab*d_model个数字组成的权重矩阵；
        super(embeddings,self).__init__() # 获得父类的成员函数；
        self.embedding = nn.Embedding(vocab,d_model)
        self.d_model = d_model
    
    def forward(self,x):
        # 之所以需要乘上维度的平方，是防止数据太小，经过后面的PE叠加后无影响了；
        return self.embedding(x) * math.sqrt(self.d_model)
    

############################ PE位置编码；############################
class Positional_Encoding(nn.Module):
    def __init__(self,d_model:int,dropout:float,len_position=500):
        super(Positional_Encoding,self).__init__()
        
        # 在训练阶段按概率p随即将输入的张量元素随机归零，常用的正则化器，用于防止网络过拟合;
        self.dropout = nn.Dropout(p=dropout)
        self.PE = torch.zeros(len_position,d_model)# 生成的PE维度为（len_position,d_model）,有d_model个不同象的正弦函数用来编码；
        # 每一个位置的对应编码为512维度的向量；

        # .arange()返回大小为(len_position)维度的向量，值为增长的step；.unsqueeze(1)在位置1处增加了一个维度；
        position = torch.arange(0.,len_position).unsqueeze(1) #维度为（5000 * 1）；
        div_term = torch.exp(torch.arange(0., d_model, 2) * (-(math.log(10000.0) / d_model)))#维度为（1 * 5000）；
         # position * div_term维度为（5000 * 1）*（1 * 512）=（5000 * 512）；
        self.PE[:,0::2] = torch.sin(position * div_term) # 偶数位置使用sin编码；
        self.PE[:,1::2] = torch.cos(position * div_term) # 基数位置使用cos编码；

        #在最初始的维度插上batch；
        self.PE = self.PE.unsqueeze(0)
        # self.register_buffer("PE",self.PE)

    def forward(self,x):
        # print("DEBUG:X.SIZE____:",np.shape(x))
        # print("DEBUG:PE.SIZE____:",np.shape(self.PE))
        x = x + Variable(self.PE[:,:x.size(1)])#转化为Variable,就不需要求导了；此时添加上了位置编码；
        return self.dropout(x)


def clones(module, N):# 用来拷贝多个层；
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

############################ 多头注意力即多层注意力机制。原先输入的词维度为512的将通过三个投影矩阵投影到更小的维度；############################
class MultiHeadedAttention(nn.Module): 
    def __init__(self, head_num:int, d_model, dropout=0.1):
        super(MultiHeadedAttention,self).__init__()
        self.d_k = d_model // head_num #d_k为输出模型大小的维度；
        self.head_num = head_num #多头的数目；
        self.dropout = dropout
        self.Linears = clones(nn.Linear(d_model,d_model),4) # 定义四个投影矩阵；
        self.Attention = None
        self.Dropout = nn.Dropout(p=dropout)
    
    def forward(self,query,key,value,mask=None):
        nbatches = query.size(0)

        seq_q_len = query.size(1)
        seq_k_len = key.size(1)
        seq_v_len = value.size(1)

        if mask is not None:
            mask = mask.unsqueeze(1) # 给mas添加一个维度，并设置其值为1；

        # 分别进行线性变换；
        # print(np.shape(query))
        query = self.Linears[0](query)
        key = self.Linears[1](key)
        value = self.Linears[2](value)

        # 重塑512维度为head_num*d_k;
        query = query.view(nbatches,seq_q_len,self.head_num,self.d_k)
        key = key.view(nbatches,seq_k_len,self.head_num,self.d_k)
        value = value.view(nbatches,seq_v_len,self.head_num,self.d_k)

        # 将与头有关的维度放在前面，方便后续注意力层进行操作；
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        # 经过注意力层，返回softmax(qk/）sqrt(d)*v;
        x,self.attn = Attention(query,key,value,mask=mask,dropout=self.Dropout)

        x = x.transpose(1,2).contiguous() # 将多头相关的维度交换回来；
        x = x.view(nbatches,-1,self.head_num * self.d_k) # 将维度重塑为512维；
        return self.Linears[-1](x) # concat掉。 

############################ 注意力层； ############################
def Attention(query,key,value,mask=None,dropout=None):
    d_k = query.size(-1) # 获取最后一个维度；
    
    # 对其进行相乘；多头的维度保持不变，使用softmax(qk/sqrt(d))*v公式进行计算;
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # print("Shape of Mask:",np.shape(mask))
    # print("Shape of Scores:",np.shape(scores))
    if mask is not None: # 如果有mask就使用，当需要制作解码器的时候使用；
        scores = scores.masked_fill_(mask == 0, -1e9) 
    p_attn = F.softmax(scores, dim=-1)  # 获取注意力分数图；

    if dropout is not None: 
        p_attn = dropout(p_attn)
    
    # 返回计算的数值和注意力分数矩阵；
    return torch.matmul(p_attn, value), p_attn 

############################ LayerNorm层； ############################
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        # features=（int）512，用来给nn.Parameter生成可训练参数的维度；
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self,x):
        # 此时的x维度为（nBatches,句子长度,512）;
        # print("X_SHAPE_IN_LN:",np.shape(x))
        mean = x.mean(-1,keepdim=True) # 对512维度的部分求均值，keepdim保持输出维度相同；
        std = x.std(-1,keepdim=True) + self.eps # 对512维度的部分求标准差，keepdim保持输出维度相同；eps保持标准差不为零；
        result = self.a_2 * (x-mean)/std + self.b_2
        # print(result.dtype)
        return result
# 有了LayerNorm层就可以构造sublayerConnection层了：
    
############################ SubLayerConnection层； ############################
class SubLayerConnection(nn.Module):
    def __init__(self,d_model,dropout=0.1):
        super(SubLayerConnection,self).__init__()
        self.LayerNorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,sublayer):
        # 采用残差结构；
        return x + self.dropout(sublayer(self.LayerNorm(x)))

############################ FFN层； ############################
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_Hidden, dropout=0.1):
        super(PositionWiseFeedForward,self).__init__()
        self.linear_1 = nn.Linear(d_model,d_Hidden)
        self.linear_2 = nn.Linear(d_Hidden,d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

######################## 所有的子层便构建完毕； ###########################
