import os
import numpy as np
import pickle
import sys 

import torch
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F

from model import DKPH
from settings import *
from utils import *

from data import get_train_loader



# sys.path.append("/home/lipd/DKPH/") #notice that the path need be changed when the filename is renamed.
if not os.path.exists(file_path):
    os.makedirs(file_path)

best_optimizer_pth_path = file_path+'/best_optimizer.pth'
optimizer_pth_path = file_path+'/optimizer.pth'
print('Learning rate: %.4f' % lr)
print('bits: %d' % nbits)
infos = {}
infos_best = {}
histories = {}

model = DKPH(feature_size).cuda()


train_loader = get_train_loader(train_feat_path, sim_path, batch_size,shuffle = True)

itera = 0
epoch = 0

optimizer = Adam(model.parameters(), lr=lr)
optim_schedule = ScheduledOptim(optimizer, hidden_size, n_warmup_steps=10000)
if os.path.exists(best_optimizer_pth_path) and use_checkpoint:
    optimizer.load_state_dict(torch.load(optimizer_pth_path))


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

total_len = len(train_loader)
criterion = torch.nn.TripletMarginLoss(margin=0.5, p=2.0, eps=1e-06)

while True: 
    decay_factor = 0.9  ** ((epoch)//lr_decay_rate)
    current_lr = max(lr * decay_factor,1e-4)
    set_lr(optimizer, current_lr) # set the decayed rate

    for i, data in  enumerate(train_loader, start=1):
        optimizer.zero_grad()

        batchsize = data["mask_input"].size(0)  
        data = {key: value.cuda() for key, value in data.items()} 
        bb1,frame1,hid1 = model.forward(data["mask_input"][:,:max_frames,:])
        bb2,frame2,hid2 = model.forward(data["mask_input"][:,max_frames:,:])
        sim = bb1.mul(bb2)         
        sim = torch.sum(sim,1)/nbits
        nei_loss = torch.sum((1*data["is_similar"].float()-sim)**2)/batchsize  
        mask_loss = (torch.sum((frame1-data["visual_word"][:,:max_frames,:])**2)\
                  +torch.sum((frame2-data["visual_word"][:,max_frames:,:])**2))\
                  /(2*max_frames*feature_size*batchsize)

        weight1=0.5*(1-data["is_similar"])
        weight1=weight1.reshape(batchsize,1)
        mu_loss = (torch.sum((torch.mean(hid1,1)-data['n1'])**2)+torch.sum((torch.mean(hid2,1)-data['n2'])**2))/(hidden_size*batchsize) + 0.1*criterion(torch.mean(hid1,1),weight1* data['n1'],weight1*data['n2'])
        loss = mask_loss +0.9*mu_loss + 0.11*nei_loss
        loss.backward()
        optimizer.step()
        
        itera += 1
        infos['iter'] = itera
        infos['epoch'] = epoch
        if itera%20 == 0 or batchsize<batch_size:  
            print 'Epoch:%d Step:[%d/%d]  neiloss: %.2f maskloss: %.2f mu_loss: %.2f' \
            % (epoch, i, total_len, nei_loss.data.cpu().numpy(),\
               mask_loss.data.cpu().numpy(),mu_loss.data.cpu().numpy())

    
    torch.save(model.state_dict(), file_path + '/dkph_16bits_1016.pth')

    torch.save(optimizer.state_dict(), optimizer_pth_path)
    if epoch>45:
        model.eval()
        evaluate(model, test_feat_path, label_path ,45600)
    with open(os.path.join(file_path, 'infos.pkl'), 'wb') as f:
        pickle.dump(infos, f)
    with open(os.path.join(file_path, 'histories.pkl'), 'wb') as f:
        pickle.dump(histories, f)
    epoch += 1
    if epoch>num_epochs:
        break
    model.train()
    
