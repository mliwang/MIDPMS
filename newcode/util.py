# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:24:48 2022

@author: 86138
"""
import pickle
import torch
import config as cf
from torch.utils.data import Dataset

class TensorDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.
    """
    def __init__(self,newsPAD,forwhat="train",istrain=True):
        super(TensorDataset, self).__init__()
        self.istrain =istrain
        if forwhat=="train":
            datadir=cf.train_pkl
        elif forwhat=="val":
            datadir = cf.val_pkl
        else:
            datadir = cf.test_pkl
        with open(datadir, 'rb') as f:
            id_data, input_seq, influce_num, user_pool, news_pool, Label_pop, Label_nextUser = pickle.load(f)
        self.id_train=id_data
        self.input_seq=input_seq
        self.influce_num=influce_num
        self.user_pool=user_pool
        self.news_pool=news_pool
        self.Label_pop=Label_pop
        self.Label_nextUser = Label_nextUser
        self.newsPAD=newsPAD

    # def getContentSim(self,index):
    #     def sim(tensor_1, tensor_2):
    #         normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    #         normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    #         return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)
    #     with open(cf.emb_dir, 'rb') as f:
    #         emVec = pickle.load(f)
    #     curNews=emVec[index]   #1,768维向量
    #     user_news=self.news_pool[index]#  用户到新闻的矩阵
    #     sim_mat=torch.zeros([len(user_news),len(user_news[0])])
    #     for j,u in enumerate(user_news):
    #
    #         for i in range(len(u)):
    #             if u[i] != self.newsPAD:
    #                 sim_mat[j][i]=sim(curNews, emVec[u[i]])
    #     # sim_mat=torch.cat([torch.cat([sim(curNews, emVec[i]) for i in u if i!=self.newsPAD else 0], dim=0) for u in user_news],dim=0)
    #     simVec=torch.sum(sim_mat,dim=1)
    #     return simVec
    def __getitem__(self, index):
        '''
        把数据处理成样本 sample:(S,X,Y)
        '''
        # simVec=self.getContentSim(index)#拿到各个候选用户目前关注的news与当前新闻的相似度

        # print("id_train:",self.id_train[index])
        # print("input_seq:",self.input_seq[index])
        # print("input_seq:", self.influce_num[index])
        if self.istrain:
            return (torch.tensor(self.id_train[index], dtype=torch.long),
                    torch.tensor(self.input_seq[index], dtype=torch.float32) ,
                    torch.tensor(self.influce_num[index], dtype=torch.float32) ,
                    torch.tensor(self.news_pool[index], dtype=torch.long),
                    torch.tensor(self.user_pool[index], dtype=torch.long) ,
                    torch.tensor(self.Label_pop[index],dtype=torch.float32),
                    torch.tensor(self.Label_nextUser[index],dtype=torch.long))

        else:
            return (torch.tensor(self.id_train[index], dtype=torch.long),
                    torch.tensor(self.input_seq[index], dtype=torch.float32) ,
                    torch.tensor(self.influce_num[index], dtype=torch.float32) ,
                    torch.tensor(self.news_pool[index], dtype=torch.long),
                    torch.tensor(self.user_pool[index], dtype=torch.long) ,
             )
    def __len__(self):
        return len(self.id_train)
