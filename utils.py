import numpy as np
import scipy.io as sio 
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import get_eval_loader
from settings import *
import time
import scipy.io as sio

import h5py


class Array():
    def __init__(self):
        pass
    def setmatrcs(self,matrics):
        self.matrics = matrics

    def concate_v(self,matrics):
        self.matrics = np.vstack((self.matrics,matrics))

    def getmatrics(self):
        return self.matrics


def evaluate(model, file_path, labels_name,num_sample):
    print 'loading test data...'
    hashcode = np.zeros((num_sample,nbits),dtype = np.float32)
    label_array = Array()

    rem = num_sample%test_batch_size
    labels = sio.loadmat(labels_name)['labels']
    eval_loader = get_eval_loader(file_path,batch_size=test_batch_size)
    label_array.setmatrcs(labels)
    
    batch_num = len(eval_loader)
    time0 = time.time()
    for i, data in enumerate(eval_loader): 
        data = {key: value.cuda() for key, value in data.items()}
        my_H,my_recon,my_2 = model.forward(data["visual_word"])
        BinaryCode = my_H 

        if i == batch_num-1:
            hashcode[i*test_batch_size:,:] = BinaryCode[:rem,:].data.cpu().numpy()
        else:
            hashcode[i*test_batch_size:(i+1)*test_batch_size,:] = BinaryCode.data.cpu().numpy()

    test_hashcode = np.matrix(hashcode)
    time1 = time.time()
    print 'retrieval costs: ',time1-time0
    Hamming_distance = 0.5*(-np.dot(test_hashcode,test_hashcode.transpose())+nbits)
    time2 = time.time()
    print 'hamming distance computation costs: ',time2-time1
    HammingRank = np.argsort(Hamming_distance, axis=0)
    time3 = time.time()
    print 'hamming ranking costs: ',time3-time2

    labels = label_array.getmatrics()
    print 'labels shape: ',labels.shape 
    sim_labels = np.dot(labels, labels.transpose())

    time6 = time.time()
    print 'similarity labels generation costs: ', time6 - time3

    records = open('./results/16.txt','w+')
    maps = []
    map_list = [5,10,20,40,60,80,100]
    for i in map_list:
        map,_,_ = mAP(sim_labels, HammingRank,i)
        maps.append(map)
        records.write('topK: '+str(i)+'\tmap: '+str(map)+'\n')
        print 'i: ',i,' map: ', map,'\n'
    time7 = time.time()
    records.close()



def save_nf(model):
    '''
    To prepare latent video features, you can first train teacher model 
    with only mask_loss and save features with this function.
    '''
    num_sample = 45585 # number of training videos
    new_feats = np.zeros((num_sample,hidden_size),dtype = np.float32)
    rem = num_sample%test_batch_size
    eval_loader = get_eval_loader(train_feat_path,batch_size=test_batch_size)   
    batch_num = len(eval_loader)
    for i, data in enumerate(eval_loader): 
        data = {key: value.cuda() for key, value in data.items()}
        _,_,x = model.forward(data["visual_word"])
        feat = torch.mean(x,1)
        if i == batch_num-1:
            new_feats[i*test_batch_size:,:] = feat[:rem,:].data.cpu().numpy()
        else:
            new_feats[i*test_batch_size:(i+1)*test_batch_size,:] = feat.data.cpu().numpy()
    h5 = h5py.File(latent_feat_path, 'w')
    h5.create_dataset('feats', data = new_feats)
    h5.close()


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

def load_data(path, dtype=np.float32):
    db = sio.loadmat(path)
    traindata = dtype(db['traindata'])
    testdata = dtype(db['testdata'])
    cateTrainTest = dtype(db['cateTrainTest'])

    mean = np.mean(traindata, axis=0)
    traindata -= mean
    testdata -= mean

    return traindata, testdata, cateTrainTest

def save_sparse_matrix(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape, _type=array.__class__)

def load_sparse_matrix(filename):
    matrix = np.load(filename)

    _type = matrix['_type']
    sparse_matrix = _type.item(0)

    return sparse_matrix((matrix['data'], matrix['indices'],
                                 matrix['indptr']), shape=matrix['shape'])

def binarize_adj(adj):
    adj[adj != 0] = 1
    return adj
        
def renormalize_adj(adj):
    rowsum = np.array(adj.sum(axis=1))
    inv = np.power(rowsum, -0.5).flatten()
    inv[np.isinf(inv)] = 0.
    zdiag = sp.diags(inv)     

    return adj.dot(zdiag).transpose().dot(zdiag)

def sign_dot(data, func):
    return np.sign(np.dot(data, func))

def mAP(cateTrainTest, IX, num_return_NN=None):
    numTrain, numTest = IX.shape

    num_return_NN = numTrain if not num_return_NN else num_return_NN

    apall = np.zeros((numTest, 1))
    yescnt_all = np.zeros((numTest, 1))
    for qid in range(numTest):
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x/(rid*1.0 + 1.0)
	yescnt_all[qid] = x
        if not p: apall[qid] = 0.0
        else: apall[qid] = p/(num_return_NN*1.0)

    return np.mean(apall),apall,yescnt_all  

def topK(cateTrainTest, HammingRank, k=500):
    numTest = cateTrainTest.shape[1]

    precision = np.zeros((numTest, 1))
    recall = np.zeros((numTest, 1))

    topk = HammingRank[:k, :]

    for qid in range(numTest):
        retrieved = topk[:, qid]
        rel = cateTrainTest[retrieved, qid]
        retrieved_relevant_num = np.sum(rel)
        real_relevant_num = np.sum(cateTrainTest[:, qid])

        precision[qid] = retrieved_relevant_num/(k*1.0)
        recall[qid] = retrieved_relevant_num/(real_relevant_num*1.0)

    return precision.mean(), recall.mean()


