import Layers as L
import Encoder_Decoder as ED
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
from torch.autograd import Variable

# ########################### 创造训练时的batch; ####################################
# class Batch:
#     def __init__(self, src, tgt=None, pad=0):
#         self.src = src # 原输入的序列；
#         self.src_mask = (src != pad).unsqueeze(-2) # 生成原输入序列的掩码；
#         if tgt is not None: 
#             # 下面的做法的目的是为了使用（src + tgt) 来预测(tgt_y)；
#             self.tgt = tgt[:, :-1] # 目标序列的前N-1个单词；
#             self.tgt_y = tgt[:, 1:] # 目标序列的后N-1个单词（相当于去掉了第一个）；
            
#             # 创造掩码，用来隐藏padding和未来的单词，以防止模型在训练时看到未来的信息。
#             self.tgt_mask = self.make_std_mask(self.tgt, pad)
            
#             # self.ntokens用来计数，来记录self.tgt_y中非填充元素的总数（只有填充元素可以计算Loss）.
#             self.ntokens = (self.tgt_y != pad).data.sum()
    
#     # 创造掩码的函数；
#     @staticmethod
#     def make_std_mask(tgt, pad):
#         tgt_mask = (tgt != pad).unsqueeze(-2)
#         final_mask = ED.subsequent_mask(tgt.size(-1))
#         final_mask = final_mask.type_as(tgt_mask.data) # 转换类型；
#         final_mask = tgt_mask & Variable(final_mask) # 使得mask不受梯度影响；
#         # 用上述方法返回的掩码为每个句子的Attention Mask（前面描述的那种）。
#         return final_mask
    

class NoamOpt: # 作者设计的一种变学习率的优化器；
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer # 优化器部分；
        self._step = 0 # 目前迭代步数；
        self.warmup = warmup # 需要进行热身的总步长步数；
        self.factor = factor # 一个自定义的幅值；
        self.model_size = model_size # 模型大小；
        self._rate = 0
        
    def step(self): # 迭代步数的记录；
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate # 修改迭代器的学习率；
        self._rate = rate # 记录当前的学习率；
        self.optimizer.step() # 进行下一步；
        
    def rate(self, step = None): #当前学习率的计算
        if step is None:
            step = self._step
        return self.factor * \
        (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    print("--build opt--")
    return NoamOpt( model.src_embed[0].d_model, 
                    2, 
                    4000,
                    torch.optim.Adam(
                        model.parameters(), 
                        lr=0, 
                        betas=(0.9, 0.98), 
                        eps=1e-9))

# plt.rcParams['font.sans-serif'] = ['SimHei']
# opt = NoamOpt(512, 1, 4000, None)
# plt.plot(np.arange(1,20000), [opt.rate(i) for i in range(1,20000)])
# plt.xlabel('迭代步长step')
# plt.ylabel('学习率lr')
# plt.show()

class LabelSmoothingKLDivLoss(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothingKLDivLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum') # KL散度损失；
        self.padding_idx = padding_idx # 对应padding部分的下标；
        self.smoothing = smoothing # 平滑值；
        self.confidence = 1.0 - smoothing # 设置真值标签的置信度 = 1 - 平滑值；
        self.size = size # 最终输出所对应的词表大小。
        self.true_dist = None # 用以存储调整后的真值的分布情况；
        print("--build loss--")
        
    # 定义Loss的前向传播过程；
    def forward(self, x, target):# x为最终模型的输出，target为目标词表；
        # print("X_SHAPE_IN_LOSS:",np.shape(x))
        # print("X_IN_LOSS:",x)
        # print("TARGET_SHAPE_IN_LOSS:",np.shape(target))        
        assert x.size(1) == self.size # x的维度为(batches,vocab)。我们默认这第二维度等于目标词表维度。
        true_dist = x.data.clone() # 深拷贝一个x。

        # fill_(k)作用是将所有的值设置为k。
        # 这里是将每个元素初始化为平滑值除以类别数减去2（为了排除最终真值的对应词的类别和padding）；
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        
        # scatter_(dim, index, src) 会根据index张量中的索引，在dim维度将src中的值填充到调用它的张量中。
        # 此处为根据 target 张量提供的索引，将置信度self.confidence填充到正确的类别位置。
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        #将padding对应位置的置信度设置为0。
        true_dist[:, self.padding_idx] = 0

        # 找出目标标签中等于padding索引的位置。
        mask = torch.nonzero(target.data == self.padding_idx)
        # print("true_dist_IN_LOSS:",true_dist)
        # 将这些位置对应的真实分布行设置为0。
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False)) # 利用真实分布去做Loss。

