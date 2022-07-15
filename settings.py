import os
import time

dataset = 'fcv'  # type=str 
workers = 8 # type=int 
batch_size = 256 #type=int 
num_epochs = 48 # type=int 
use_cuda = True
use_checkpoint = False
lr = 3e-4 #type=float 
lr_decay_rate = 20 #type=float 
single_lr_decay_rate = 20 #type=float 

weight_decay = 1e-4 #type=float 

nbits = 16
feature_size = 4096
max_frames = 25
hidden_size = 256

test_batch_size = 256
nnachors = 2000

data_root = '/home2/lipd/fcv/'  #to save large data files
home_root = '/home/lipd/DKPH/'
sim_path = data_root+'sim_matrix.h5'

train_feat_path = data_root+'fcv_train_feats.h5'
test_feat_path = data_root+'fcv_test_feats.h5'
label_path = data_root+'fcv_test_labels.mat'


train_assist_path = home_root+'data/train_assit.h5'  #refer BTH
latent_feat_path = home_root+'data/latent_feats.h5' #refer BTH
anchor_path = home_root+'data/anchors.h5' #refer BTH
save_dir = home_root+'models/' + dataset
file_path = save_dir + '_bits_' + str(nbits)
