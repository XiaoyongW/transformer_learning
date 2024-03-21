import Layers as L
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy

######################## 构建单个Encoder块； ###########################
class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        # d_model是输入维度数据；
        super(EncoderLayer,self).__init__()
        self.self_attn = self_attn #自注意力层；
        self.feed_forward = feed_forward # FFN层；
        self.sublayer = L.clones(L.SubLayerConnection(d_model,dropout),2) # 两个残差链接结构；
        self.d_model = d_model
    
    def forward(self, x, mask=None):
        # 第一个残差结构，通过自注意力机制输入；其中经过了多头注意力层 + LN层，并通过残差结构连接；
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) 
        # 第二个残差结构，通过FFN；其中经过了FFN层，并通过残差结构连接；
        x = self.sublayer[1](x, self.feed_forward) 
        return x


######################## 整个Encoder； ###########################
class Encoder(nn.Module):
    def __init__(self, EncodeLayer, num_of_EncodeLayer:int):
        # d_model是输入维度数据；
        super(Encoder, self).__init__()
        self.EncodeLayers = L.clones(EncodeLayer,num_of_EncodeLayer)
        self.LayerNorm = L.LayerNorm(EncodeLayer.d_model) # 为d_model维；
    
    def forward(self, x, mask=None):
        for encodelayer in self.EncodeLayers:
            x = encodelayer(x, mask)
            # print("X_SHAPE_IN_ENCODER:",np.shape(x))
            x = self.LayerNorm(x)
        return x

######################## 构建单个Decoder块； ###########################
class DecoderLayer(nn.Module): # 单独一个Decoder层；
    def __init__(self, d_model, self_attn, src_attn, 
      feed_forward, dropout):
      # size = d_model=512;
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = L.clones(L.SubLayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # 来自Encoder的query与value:用作解码序列的query与value。
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # print("M_SHAPE_IN_DECODER:",np.shape(m))
        # print("X_SHAPE_IN_DECODER:",np.shape(x))
        x = self.sublayer[1](x, lambda x: self.self_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

######################## 构建整个Decoder； ###########################
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.decodelayers = L.clones(layer, N)
        self.layerNorm = L.LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.decodelayers:
            x = layer(x, memory, src_mask, tgt_mask)
            x = self.layerNorm(x)
        return x
    
######################## 构建整个输出头； ###########################
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.output = nn.Linear(d_model, vocab) # 输入是d_model的维度,输出是词表大小；

    def forward(self, x):
        return F.log_softmax(self.output(x), dim=-1) # 使用softmax修正概率；
        # return F.softmax(self.output(x), dim=-1) # 使用softmax修正概率；
    

######################## 构建整个大模型的编码-解码结构； ###########################
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # 最开始经过词嵌入和PE模块的编码序列；
        self.tgt_embed = tgt_embed # 最后结果的经过词嵌入和PE模块的编码序列；
        self.generator = generator # 输出头；

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.src_embed(src) # 序列经过词嵌入层和PE编码；
        src = self.encoder(src, src_mask) # 序列经过编码器；
        embed_tgt = self.tgt_embed(tgt) # 经过词嵌入层的tgt;
         # src_mask是第一层多头注意力机制的mask,tgt_mask是第二层多头注意力机制的mask；
        Decoder_result = self.decoder(embed_tgt, src, src_mask, tgt_mask)
        output = self.generator(Decoder_result)
        # print("output_SHAPE_IN_LOSS:",np.shape(output))
        return output
    
######################## 构建mask的上三角阵； ###########################
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size) # (1, 10, 10)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # triu生成一个三角矩阵，k对应的维度的对角线往下的元素均为0，上三角的元素均为1； 
    return torch.from_numpy(subsequent_mask) == 0 #反转矩阵：让元素反着来；

######################## 构建模型； ###########################
def make_model(src_vocab, tgt_vocab, N=2, d_model=512, d_ff=2048, h=8, dropout=0.1):
    # src_vocab = 源语言词表大小；
    # tgt_vocab = 目标语言词表大小；
    # d_model = 经过embedding后的扩充维度大小；
    # d_ff = 在ffn时候的隐藏层大小；
    # h = 头大小；
    c = copy.deepcopy # 深度，这里使用deepcopy防止因指针指向同一片地方而导致发生的干扰。
    attn = L.MultiHeadedAttention(h, d_model) #多头注意力层；
    ff = L.PositionWiseFeedForward(d_model, d_ff, dropout) # FFN层；
    position = L.Positional_Encoding(d_model, dropout) # 位置编码层；

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(L.embeddings(d_model, src_vocab), c(position)), # nn.Sequential实现模型的顺序连接；
        nn.Sequential(L.embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)) #调用__init__()进行初始化；

    # 对所有层参数使用Xavier初始化；
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    print("--build model--")
    return model