import os
import numpy as np
import pickle
from collections import OrderedDict
import pdb 

seqlist_path = '/home/wangxiao/Documents/rgb_event_tracking_benchmark/visEvent_dataset/VisEvent_train_subset.txt'
output_path  = '/home/wangxiao/Documents/rgb_event_tracking_benchmark/visEvent_dataset/VisEvent_train_subset_mdnet.pkl'
set_type = 'VisEvent'
seq_home = '/home/wangxiao/Documents/rgb_event_tracking_benchmark/visEvent_dataset/train/'

with open(seqlist_path,'r') as fp:
    seq_list = fp.read().splitlines()

data = {}
for i,seqname in enumerate(seq_list):
    print(seqname)

    if set_type=='VisEvent':
        seq_path = seq_home + seqname
        vis_img_list   = sorted([p for p in os.listdir(seq_path+'/vis_imgs') if os.path.splitext(p)[1] == '.bmp'])
        event_img_list = sorted([p for p in os.listdir(seq_path+'/event_imgs') if os.path.splitext(p)[1] == '.bmp'])
        gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')

    assert len(vis_img_list) == len(gt), "vis Lengths do not match!!" 
    assert len(event_img_list) == len(gt), "vis Lengths do not match!!"

    if gt.shape[1]==8:
        x_min = np.min(gt[:,[0,2,4,6]],axis=1)[:,None]
        y_min = np.min(gt[:,[1,3,5,7]],axis=1)[:,None]
        x_max = np.max(gt[:,[0,2,4,6]],axis=1)[:,None]
        y_max = np.max(gt[:,[1,3,5,7]],axis=1)[:,None]
        gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

    # pdb.set_trace() 
    vis_img_list_full = [] 
    event_img_list_full = [] 
    for ii in range(len(vis_img_list)): 
        line = vis_img_list[ii] 
        imagePath = seq_path+'/vis_imgs/'+line 
        vis_img_list_full.append(imagePath) 

        line = event_img_list[ii] 
        imagePath = seq_path+'/event_imgs/'+line 
        event_img_list_full.append(imagePath) 

    data[seqname] = {'vis_images':vis_img_list_full, 'event_images':event_img_list_full, 'gt':gt}

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)
