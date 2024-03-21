import torch
import Dataset as Data
from nltk.tokenize import word_tokenize
import Encoder_Decoder as ED


# 因为数据量比较小，因此我们直接生成一个词表就行了！
DataLoader,src_vocab,tgt_vocab = Data.BuildDataLoader(1000)

# 用于测试的翻译句子：
test_sentence = 'Officials underestimated tail risks.'

# 将他转化为数值化后模型可以输入的样子：
test_tokens = [word_tokenize(test_sentence.lower())]
test_src = Data.numericalize(test_tokens, src_vocab)
tgt = torch.LongTensor([[1]]) #只有一个<sos>；

# 加载模型；
model = ED.make_model(src_vocab=len(src_vocab),# 源词表大小；
                            tgt_vocab=len(tgt_vocab),# 目标词表大小；
                            N=2, # Encoder层和Decoder层各两层；
                            d_model=512,
                            d_ff=2048, # FFN层；
                            h=8) # 多头；
model.load_state_dict(torch.load('model.pt'))
model.eval()


max_len = 10
mask_shape = 1
for i in range(max_len):
    Mask = ED.subsequent_mask(mask_shape)
    output = model(test_src,tgt,None,Mask) # 获得模型的输出；
    result = torch.argmax(output[:,-1]) # 找到最后一个字预测概率最大的索引；
    tgt = torch.concat([tgt,result.unsqueeze(0).unsqueeze(0)],dim=1) # 将他们拼接起来；
    mask_shape += 1 #增加mask的尺寸；
    if(result == torch.tensor(2)): # 如果是<eos>，说明解码结束，跳出。
        break


# 定义一个从词表中寻找我们所需要的文字的函数；
def getkey(vocab,value_need):
    key = [key for key, value in vocab.items() if value == value_need]
    return key

# 根据词表输出中文。
chi_result = ""
for i in range(len(tgt[0])):
    key = int(tgt[0][i])
    word = getkey(tgt_vocab,key)[0]
    chi_result += word

print("原始句子：",test_sentence)
print(chi_result)