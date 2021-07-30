import torch
from pytorch_transformers import *
import logging
import argparse
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
import math
import numpy
import collections
import random
from tqdm import tqdm
import pickle
from opts import parser


def get_candi_event(img_val):
  
  positive_list = list()
  negative_list = list()
  
  for tmp_event_pair in img_val:
      if tmp_event_pair[1] == 1:
          positive_list.append(tmp_event_pair)
      if tmp_event_pair[1] == 0:
          negative_list.append(tmp_event_pair)
                      
  random.shuffle(negative_list)
  negative_list = negative_list[:len(positive_list)] ## add negative same amount of positive
  positive_list.extend(negative_list) # add negative in candidate list 
  
  candidate_list = positive_list
  print('candi_list', candidate_list )
  
  return candidate_list

for tmp_event_pair in candidate_list:
            event_1 = tmp_event_pair[0].split('$$')[0]
            event_2 = tmp_event_pair[0].split('$$')[1]
         
if __name__ == '__main__' :
  dataset_path = './data/Contextual_dataset.json'
  
  with open(dataset_path, 'r') as f:
      initial_dataset_by_video = json.load(f)
      
  for tmp_video_id in initial_dataset_by_video:
    video = initial_dataset_by_video[tmp_video_id]
    print(f'video id : {video} \n\n')
    
    img0_event = video['image_0']['event'].values() 
    get_candi_event(img0_event)
    
    img1_event = video['image_1']['event'].values()
    get_candi_event(img1_event)
    
    img2_event = video['image_2']['event'].values() 
    get_candi_event(img2_event)
    
    img3_event = video['image_3']['event'].values()
    get_candi_event(img3_event)
    
    img4_event = video['image_4']['event'].values() 
    get_candi_event(img4_event)
    
                