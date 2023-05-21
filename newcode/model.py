
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import pickle
def get_pad_mask(seq, pad_idx=0):
    return (seq != pad_idx)#.unsqueeze(-2)
class MSMfusion(nn.Module):
    def __init__(self,user_num,influence_dim=3,input_feature_len=1,bidirectional=True,hidden_size=128, sequence_len=64,rnn_num_layers=5,
                 rnn_dropout=0.1,emdir="../data/aminer/process/bert.pkl", out=1, dropout=0.1,device="cpu",pre_train=True,
                 n_head=2,padding_idx=0,opt=None):
        super(MSMfusion, self).__init__()
        self.use_att=opt.use_att
        if pre_train:
            with open(emdir, 'rb') as f:
                emVec = pickle.load(f)
                # del emVec[opt.userPAD]
                emVec = torch.tensor(emVec.reshape(len(emVec), -1), dtype=torch.float32)
            self.em=nn.Embedding.from_pretrained(emVec, freeze=False,padding_idx=padding_idx)
        else:
            self.em = nn.Embedding(opt.content_num,hidden_size,padding_idx=padding_idx)
        if opt.use_att:
            from layers import MultiHeadAttention
            self.slf_attn = MultiHeadAttention(n_head, opt.news_size, opt.news_size, opt.news_size, dropout=dropout)

        self.gru = nn.GRU(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout
        )
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.grulinear=nn.Linear(hidden_size, 1)
        self.timefactor = nn.Linear(hidden_size, 1)
        self.norm1=nn.BatchNorm1d(hidden_size)
        self.device=device
        self.user_num=user_num

        self.rnn_directions = 2 if bidirectional else 1

        self.Anticipation = nn.Linear(influence_dim, hidden_size)
        self.norm2 = nn.BatchNorm1d(hidden_size+user_num+1)
        self.linearmsm1 = nn.Linear(hidden_size+user_num+1, hidden_size)
        self.norm2 = nn.BatchNorm1d(hidden_size)
        self.linearmsm2 = nn.Linear(hidden_size, 1)

        # self.linear2 = nn.Linear(hidden_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    def init_weights(self):
        init.xavier_normal_(self.grulinear.weight)
        init.xavier_normal_(self.timefactor.weight)
        init.xavier_normal_(self.Anticipation.weight)
        init.xavier_normal_(self.linearmsm1.weight)
        init.xavier_normal_(self.linearmsm2.weight)

    def getContentSim(self,user2newsDict, index):
        curNews = self.em(index)  # b, content dim
        # user_news = self.news_pool[index]  # 用户到新闻的矩阵
        # print(user2newsDict)
        # print("看看",self.em.size())

        attentionnewsF=self.em(user2newsDict)
        batch, userlen, newslen, dim = attentionnewsF.size()
        curNews = curNews.repeat(1, userlen).reshape(batch, userlen, dim).unsqueeze(2).to(self.device)
        simi = torch.cosine_similarity(attentionnewsF, curNews, -1)#batch, userlen, newslen
        mask=get_pad_mask(user2newsDict)
        # print("user2newsDict",user2newsDict.size())


        if self.use_att:
            #用自注意力机制找到当前user具体关注哪些news

            simi, _=self.slf_attn(simi,simi,simi)#enc_slf_attn注意力系数
        simi = simi.masked_fill(mask == 0, -1e9)
        simVec = torch.sum(simi, dim=-1)
        return simVec
    def forward(self, input_seq,influence_num, user2newsDict,idx):
        '''
        input_seq,   流行度序列
        influence_num, 当前新闻的转发数、评论数
        sim_vec,  当前新闻和各个2-hop news的相似度
        user2newsDict   各个候选user原始关注的news的id
        return
        pop_t 下一个时刻该新闻的流行度
        infected_users  下一个时刻该新闻的影响的用户
        '''
        #1.对新闻的流行度进行编码和预测
        gru_out, hidden=self.popModel(input_seq)#hidden就表示流行度变化的时间因子，gru_out为对最后一个时刻的流行度的预测  hidden: b,hidden_size
        pop_gru=self.grulinear(self.norm1(gru_out))
        pop_gru=F.relu(pop_gru)
        hidden=F.softplus(self.timefactor(hidden))# b,1

        #2.对根据新闻内容计算新闻之间的相似关系  只算当前新闻与候选用户相关的内容之间的相似度


        sim_vec=self.getContentSim(user2newsDict, idx)#user2newsDict b,user_pool_size,news_size    idx: b
        sim_vec=F.softmax(sim_vec,dim=1)#对候选池中所有的user的相似度做sofmax运算保证相似度在0-1范围内  b,user_num
        #
        # print("influence_num",influence_num.size())
        # 3.学习用户新闻拓扑结构获取网络舆情初始影响力
        h=self.Anticipation(influence_num) #b,hidden_size
        out_msm=torch.cat([hidden,sim_vec,h],dim=1)
        pop_t=self.linearmsm1(out_msm)
        pop_t=self.norm2(pop_t)
        pop_t =self.linearmsm2(pop_t)
        pop_t=F.softplus(pop_t)
        # print(pop_gru,pop_t)
        pop_t=(pop_gru+pop_t)/2
        # pop_t =F.relu(pop_t)#b,1
        # pop_t=torch.nn.functional.log_softmax(pop_t, dim=1)

        #4.基于协同过滤方法实现可能影响的用户排序  根据极简演替模型计算当前新闻被各个候选用户目前关注的新闻代替的概率    p_(j,u)=sum(Π_(i→j))=sum(lambda_ij *N_j/t_j)=N_j/t_j *sum_ij{sim_ij}
        pro=pop_t / hidden * sim_vec
        # pro=F.softmax(pro,dim=1)
        pop_t = F.relu(pop_t)

        return pop_t, pro

    def popModel(self,input_seq):
        '''
        对新闻的流行度进行编码和预测
        '''
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size).to(self.device)
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        # print(input_seq.shape)
        gru_out, hidden = self.gru(input_seq, ht)
        # print(gru_out.shape)#[64, 201, 512]
        # print(hidden.shape)#[20, 64, 256]
        if self.rnn_directions * self.num_layers > 1:
            num_layers = self.rnn_directions * self.num_layers
            if self.rnn_directions > 1:
                gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
                gru_out = torch.sum(gru_out, axis=2)
            hidden = hidden.view(self.num_layers, self.rnn_directions, input_seq.size(0), self.hidden_size)
            if self.num_layers > 0:
                hidden = hidden[-1]
            else:
                hidden = hidden.squeeze(0)
            hidden = hidden.sum(axis=0)
        else:
            hidden.squeeze_(0)
        gru_out=gru_out[:,-1,:]
        return gru_out, hidden

