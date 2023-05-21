import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from Optim import ScheduledOptim
from model import MSMfusion
from util import TensorDataset
from torch.utils.data import Dataset,DataLoader,RandomSampler,SequentialSampler
import gc
import torch.nn.functional as F
import config as cf
from Metrics import Metrics
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('-data_name', default='aminer')
parser.add_argument('-epoch', type=int, default=300)
parser.add_argument('-warmup', type=int, default=80)
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-d_model', type=int, default=256)
parser.add_argument('-newsPAD', type=int, default=15715)
parser.add_argument('-userPAD', type=int, default=38972)
parser.add_argument('-initialFeatureSize', type=int, default=256)
parser.add_argument('-n_warmup_steps', type=int, default=100)
parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-log', default=None)
parser.add_argument('-save_path', default= "checkpoint/MSMDiffusionPrediction.pt")
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('-no_cuda',  type=bool, default=False)
parser.add_argument('-pre_train', type=bool, default=True)
parser.add_argument('-pos_emb', type=bool, default=True)
parser.add_argument('-use_att', type=bool, default=False)
opt = parser.parse_args()
NewsPAD, UserPAD = pickle.load(open(cf.information, 'rb'))
opt.userPAD=UserPAD
opt.newsPAD=NewsPAD
opt.news_size=cf.news_size
opt.data_name=cf.data_name
opt.save_path="checkpoint/MSMDiffusionPrediction_"+cf.data_name+".pt"
if opt.pre_train:
    opt.emb_dir=cf.emb_dir
else:
    #不预训练，给指定内容数和特征维数
    with open(cf.news_dict, 'rb') as f:
        o,_ = pickle.load(f)
    opt.content_num=len(o)+1

metric = Metrics(opt.userPAD)
if opt.no_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
def get_performance(crit, pred, gold,candidates):
    # print("pred", pred)
    # print("gold", gold)

    loss = crit(pred, gold)
    pred=F.softmax(pred,dim=1)
    pred = pred.max(1)[1]
    # pred=candidates.gather(1, pred.max(1)[1].unsqueeze(0)).squeeze(0)
    # gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(opt.userPAD).data).sum().float()
    # print("loss",loss)
    return loss, n_correct
def finaltest_epoch(model, validation_data, k_list=[1, 5, 10,20,50,100]):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    all_mlse=[]
    pred_pop=[]
    pred_next=[]
    y=[]
    label_next=[]
    with torch.no_grad():
        for i, batch in enumerate(
                validation_data):  # tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
            # print("Validation batch ", i)
            # prepare data
            id_x,input_seq,influce_num,newspool,candidates,y_pop,y_next =(x.to(device) for x in batch)
            # forward
            pop_t, pro=model(input_seq, influce_num, newspool,id_x)#pro最后没有归一化，用来计算概率
            pred = F.sigmoid(pro)
            pred = pred.topk(max(k_list), dim=1)[1]
            # pred = candidates.gather(1, pred.max(1)[1].unsqueeze(0)).squeeze(0)
            msle=torch.mean(torch.pow(pop_t-y_pop,2)).cpu().numpy()
            all_mlse.append(msle)

            scores_batch, scores_len = metric.compute_metric(pred, y_next.cpu().numpy(), k_list)
            n_total_words += scores_len

            pred_pop.append(pop_t.cpu().numpy())
            pred_next.append(pred.cpu().numpy())
            y.append(y_pop.cpu().numpy())
            label_next.append(y_next.cpu().numpy())


            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    if opt.use_att:
        namedir="FinalPred_%s.pkl"%(opt.data_name)
    else:
        namedir = "FinalPred_%s_withoutATT.pkl" % (opt.data_name)

    pickle.dump((np.stack(pred_pop,0),np.stack(y,0),np.stack(pred_next,0),np.stack(label_next,0)), open(namedir,'wb'))
    return scores,np.mean(all_mlse)


def test_epoch(model, validation_data, k_list=[1, 5, 10,20,50,100]):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    all_mlse=[]
    with torch.no_grad():
        for i, batch in enumerate(
                validation_data):  # tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
            # print("Validation batch ", i)
            # prepare data
            id_x,input_seq,influce_num,newspool,candidates,y_pop,y_next =(x.to(device) for x in batch)
            # forward
            pop_t, pro=model(input_seq, influce_num, newspool,id_x)#pro最后没有归一化，用来计算概率
            pred = F.sigmoid(pro)
            pred = pred.topk(max(k_list), dim=1)[1]
            # pred = candidates.gather(1, pred.max(1)[1].unsqueeze(0)).squeeze(0)
            # print("流行度的预测：",pop_t,y_pop)
            msle=torch.mean(torch.pow(pop_t-y_pop,2)).cpu().numpy()
            all_mlse.append(msle)

            scores_batch, scores_len = metric.compute_metric(pred, y_next.cpu().numpy(), k_list)
            n_total_words += scores_len
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores,np.mean(all_mlse)
def train_model(model, train_loader,val_loader,test_loader,display_step=20):
    gc.collect()
    print(f"make_data ok!!!, epoches = {opt.epoch}")
    loss_func = nn.CrossEntropyLoss(size_average=False)#算下一个用户预测的损失
    loss_pop=nn.MSELoss()
    optimizerAdam = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps)

    if torch.cuda.is_available() and opt.no_cuda:
        model = model.to(device)
        loss_func = loss_func.to(device)
        loss_pop=loss_pop.to(device)

    validation_history = 0.0
    best_scores = {}
    best_MLSE=10
    if opt.use_att:
        namedir="result-%s.txt"%(opt.data_name)
    else:
        namedir ="result-%s_withoutATT.txt"%(opt.data_name)
    fo=open(namedir,"w")
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')
        start = time.time()
        model.train()
        train_loss = 0
        n_total_correct = 0.0
        step=0
        for batch in tqdm(
                train_loader, mininterval=2,
                desc='  - (Training)   ', leave=False):
            optimizer.zero_grad()
            # prepare data
            id_x,input_seq,influce_num,newspool,candidates,y_pop,y_next =(x.to(device) for x in batch)
            pop_t, pro=model(input_seq, influce_num, newspool,id_x)
            l1,n_correct=get_performance(loss_func, pro,y_next,candidates)
            if step>opt.warmup:
                l2=loss_pop(pop_t,y_pop)
                loss =l1 + l2
            else:
                loss = l1

            loss.backward()
            n_total_correct += n_correct
            train_loss += loss.item()
            step = step + 1
            # if step%display_step==0:
            #     print('  - (Step: {ste:d})   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
            #           'elapse: {elapse:3.3f} min'.format(
            #         ste=step,
            #         loss=loss, accu=100 * (n_total_correct/step),
            #         elapse=(time.time() - start) / 60))

            optimizer.step()
        train_loss /= len(train_loader.dataset)
        train_accu=n_total_correct/len(train_loader.dataset)
        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu,
            elapse=(time.time() - start) / 60))

        if (epoch_i + 1) % 2 == 0:
            start = time.time()
            scores,mlse = test_epoch(model, val_loader)
            if mlse<best_MLSE:
                best_MLSE=mlse
            fo.write("Epoch:%d ************"%(epoch_i))
            fo.write('\r\n')
            print('  - ( Validation )) ')
            fo.write('  - ( Validation )) \r\n')

            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
                fo.write(metric + ' ' + str(scores[metric])+' \r\n')
            print('  - ( Validation )) MSLE:',mlse)
            print("Validation use time: ", (time.time() - start) / 60, "min")
            fo.write('  - ( Validation )) MSLE:{}'.format(mlse)+" \r\n")
            fo.write("Validation use time: %f"%((time.time() - start) / 60)+ "min"+" \r\n")



            if validation_history <= sum(scores.values()):
                print("Best Validation hit@10:{} at Epoch:{}".format(scores["hits@10"], epoch_i))
                print("Best MLSE: ", best_MLSE)
                fo.write("Best Validation hit@10:{} at Epoch:{}  /n".format(scores["hits@10"], epoch_i))
                validation_history = sum(scores.values())
                best_scores = scores
                print("Save best model!!!")
                torch.save(model.state_dict(), opt.save_path)
    print('  - (final Test) ')
    scores,mlse = finaltest_epoch(model, test_loader)
    if mlse < best_MLSE:
        best_MLSE = mlse
    for metric in scores.keys():
        print(metric + ' ' + str(scores[metric]))
    print('  - ( Test )) MSLE:', mlse)
    print(" -(Finished!!) \n Best scores: ")
    fo.write(" -(Finished!!) \n Best scores:  \r\n")
    for metric in best_scores.keys():
        print(metric + ' ' + str(best_scores[metric]))
        fo.write(metric + ' ' + str(best_scores[metric])+' \r\n')
    print("Best MLSE: ",best_MLSE)
    fo.write("Best MLSE: "+str(best_MLSE) + ' \r\n')
    fo.close()

def main():
    '''
    '''
    # ========= set parameters =========#
    print("test on",opt.data_name)
    # ========= Preparing DataLoader =========#
    train = TensorDataset(opt.newsPAD,forwhat="train")
    valid = TensorDataset(opt.newsPAD,forwhat="val")
    test = TensorDataset(opt.newsPAD,forwhat="test")
    train_loader = DataLoader(train,shuffle=True, batch_size=opt.batch_size, num_workers=0)
    val_loader = DataLoader(valid,batch_size=4,shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
    # ========= Preparing Model =========#
    model = MSMfusion(cf.userpool_size,influence_dim=cf.influence_dim, sequence_len=cf.squenceLen, dropout=opt.dropout,device=device,opt=opt)
    train_model(model, train_loader,val_loader,test_loader)

    return
if __name__ == '__main__':
    main()
