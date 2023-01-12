from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import jupyter
import matplotlib
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

#定义翻译助手类
class Lang:
    def __init__(self, name):
        self.name = name
        #单词->索引
        self.word2index = {}
        #每个单词计数
        self.word2count = {}
        #索引->单词
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    #添加语句
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    #添加单词
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

#转换语句格式，将unicode变成ascii
def unicodeToAscii(s):
    return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
    )


# 其中normalizeString函数中的正则表达式需对应更改，否则会将中文单词替换成空格
def normalizeString(s):
    #变成小写，去掉前后空格
    s = s.lower().strip()
    if ' ' not in s:
        s = list(s)
        s = ' '.join(s)
    s = unicodeToAscii(s)  #将unicode变成ascii
    s = re.sub(r"([.。!！?？])", "", s)
    return s

#读入翻译数据集
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    file_path = "./data/eng-cmn.txt"
    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]

    # Reverse pairs, make Lang instances
    #reverse=true中翻英
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    #英翻中
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

#语句最大长度为10
MAX_LENGTH = 10

#英文语句前缀
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

#简化英文语句
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

#简化英文语句
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

#数据准备，数据导入函数
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)#简化每一条语句
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

#中翻英
input_lang, output_lang, pairs = prepareData('eng', 'cmn', True)
print(random.choice(pairs))

#定义编码器
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        #激活函数Embedding
        self.embedding = nn.Embedding(input_size, hidden_size)
        #激活函数GRU
        self.gru = nn.GRU(hidden_size, hidden_size)

    #函数值预测
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    #初始化隐藏层
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#定义基于注意力机制的解码器
class AttnDecoderRNN(nn.Module):
    #初始化函数
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        #翻译时的最长长度
        self.max_length = max_length

        #激活函数层+线性层
        #激活函数Embedding
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        #线性层Linear
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        #混合线性层Linear
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        #Dropout层
        self.dropout = nn.Dropout(self.dropout_p)
        #GRU层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        #线性输出层
        self.out = nn.Linear(self.hidden_size, self.output_size)

    #函数预测
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    #初始化隐藏层
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#准备训练数据
#获取语句中某个单词的索引
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

#将单个语句转换成张量
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

#从翻译对中转换成张量
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

#指使用实际目标输出作为下一个输入，
# 而不是使用解码器的猜测作为下一输入。
# 使用teacher_forcing_ratio使其收敛得更快，
# 但当训练的网络被利用时，它可能会表现出不稳定性。
teacher_forcing_ratio = 0.5

#单次模型迭代
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    #初始化隐藏层
    encoder_hidden = encoder.initHidden()

    #将两个模型的梯度归0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #获取输入层和下一次输入层的长度
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    #初始化编码器输出
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    #获取编码器输出层数据和隐藏层数据
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    #获取解码器输入
    decoder_input = torch.tensor([[SOS_token]], device=device)

    #解码器隐藏层为编码器隐藏层
    decoder_hidden = encoder_hidden

    #加速收敛
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        #若有teacher forcing，则将target层数据作为下一次输入
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        #若无teacher forcing，则将预测值作为下一个输入
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    #反向传播计算梯度
    loss.backward()

    #参数优化
    encoder_optimizer.step()
    decoder_optimizer.step()

    #返回损失
    return loss.item() / target_length

#This is a helper function to print time elapsed and estimated time remaining given the current time and progress %.

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#辅助函数，用于打印给定当前时间和进度%的已用时间和估计剩余时间
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

#开始迭代
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()#启动计时器
    plot_losses = [] #保存损失
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    #定义SGD参数优化方法
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    #定义损失函数
    criterion = nn.NLLLoss()

    #迭代开始
    for iter in range(1, n_iters + 1):
        #获取训练翻译对
        training_pair = training_pairs[iter - 1]
        #获取输入张量
        input_tensor = training_pair[0]
        #获取中间层张量
        target_tensor = training_pair[1]
        #一次训练
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    #训练结果绘图
    showPlot(plot_losses)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

#定义绘图函数
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

#模型评估
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    #取消梯度下降
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        #获取编码器输出
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        #获取模型预测结果
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=100):
    sum_scores = 0
    for i in range(n):
        #随机从训练数据集中获取测试集
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        #模型预测
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        w = []
        words = pair[1].strip(' ').split(' ')
        words.append('<EOS>')
        w.append(words)
        bleu_score = sentence_bleu(w, output_words)
        sum_scores += bleu_score
    print('The bleu_score is ', sum_scores/n)

hidden_size = 256
#定义对象
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
#模型训练
trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
#模型评估
evaluateRandomly(encoder1, attn_decoder1)

#结果可视化
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

output_words, attentions = evaluate(
    encoder1, attn_decoder1, "你 只 是 玩")
print(output_words)
plt.matshow(attentions.numpy())
plt.show()

#展示测试结果以及数据可视化
def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

#模型预测函数
def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


#模型预测
evaluateAndShowAttention("他 和 他 的 邻 居 相 处 ")

evaluateAndShowAttention("我 肯 定 他 会 成 功 的 ")

evaluateAndShowAttention("他 總 是 忘 記 事 情")

evaluateAndShowAttention("我 们 非 常 需 要 食 物 ")