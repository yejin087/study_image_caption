import json
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

class DataLoader:
  def __init__(self,device):
    self.k = 10
    self.dataset_path = 'Contextual_dataset.json'
    self.glove_path = 'glove.txt'        
    self.word_embeddings = self.load_embedding_dict(self.glove_path) # glove path
    self.device = device
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open(self.dataset_path, 'r') as f:
      self.raw_dataset = json.load(f)
  
  def load_train(self):
    train_set = self.tensorize_example(self.raw_dataset ['training'], 'train')
    print('successfully loaded %d examples for training data' % len(train_set))
    
    return train_set
    
  def load_dev(self):     
    dev_set = self.tensorize_example(self.raw_dataset ['validation'], 'dev')
    print('successfully loaded %d examples for dev data' % len(dev_set))
    
    return dev_set
    
  def load_test(self):
    test_set = self.tensorize_example(self.raw_dataset ['testing'], 'test')
    print('successfully loaded %d examples for test data' % len(test_set))
    return test_set
  
  def load_embedding_dict(self, path):
    print("Loading word embeddings from {}...".format(path))
    default_embedding = numpy.zeros(300)
    embedding_dict = collections.defaultdict(lambda: default_embedding)
    if len(path) > 0:
        vocab_size = None
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                word_end = line.find(" ")
                word = line[:word_end]
                embedding = numpy.fromstring(line[word_end + 1:], numpy.float32, sep=" ")
                assert len(embedding) == 300
                embedding_dict[word] = embedding
        if vocab_size is not None:
            assert vocab_size == len(embedding_dict)
        print("Done loading word embeddings.")
    return embedding_dict
  
  def tensorize_example(self, initial_dataset_by_video, mode):
    if mode == 'train':
        tensorized_dataset = list()
        for tmp_video_id in initial_dataset_by_video:
            tmp_video = initial_dataset_by_video[tmp_video_id]
            # if event exists
            if len(tmp_video['image_0']['event']) > 0:
                tensorized_dataset += self.train_tensorize_frame(tmp_video['image_0'], tmp_video['image_1'],
                                                           tmp_video['category'], tmp_video_id.split('_')[1], int(0))
            if len(tmp_video['image_1']['event']) > 0:
                tensorized_dataset += self.train_tensorize_frame(tmp_video['image_1'], tmp_video['image_2'],
                                                           tmp_video['category'], tmp_video_id.split('_')[1], int(1))
            if len(tmp_video['image_2']['event']) > 0:
                tensorized_dataset += self.train_tensorize_frame(tmp_video['image_2'], tmp_video['image_3'],
                                                           tmp_video['category'], tmp_video_id.split('_')[1], int(2))
            if len(tmp_video['image_3']['event']) > 0:
                tensorized_dataset += self.train_tensorize_frame(tmp_video['image_3'], tmp_video['image_4'],
                                                           tmp_video['category'], tmp_video_id.split('_')[1], int(3))
                                                           
    if mode == 'test' or mode=='dev':
        tensorized_dataset = list()
        for tmp_video_id in initial_dataset_by_video:
            tmp_video = initial_dataset_by_video[tmp_video_id]
            # if event exists
            if len(tmp_video['image_0']['event']) > 0:
                tensorized_dataset += self.test_tensorize_frame(tmp_video['image_0'], tmp_video['image_1'],
                                                           tmp_video['category'], tmp_video_id.split('_')[1], int(0))
            if len(tmp_video['image_1']['event']) > 0:
                tensorized_dataset += self.test_tensorize_frame(tmp_video['image_1'], tmp_video['image_2'],
                                                           tmp_video['category'], tmp_video_id.split('_')[1], int(1))
            if len(tmp_video['image_2']['event']) > 0:
                tensorized_dataset += self.test_tensorize_frame(tmp_video['image_2'], tmp_video['image_3'],
                                                           tmp_video['category'], tmp_video_id.split('_')[1], int(2))
            if len(tmp_video['image_3']['event']) > 0:
                tensorized_dataset += self.test_tensorize_frame(tmp_video['image_3'], tmp_video['image_4'],
                                                           tmp_video['category'], tmp_video_id.split('_')[1], int(3))
    return tensorized_dataset
  
  def test_tensorize_frame(self, image_one, image_two, category, video_id, image_id):
  
    # add event 1,2 entities confidence and select top k entities from sorted list
    all_entities = dict()
    for tmp_entity in image_one['entity']:
        #print ('tensorize_fram  : {}'.format(tmp_entity))
        if tmp_entity not in all_entities:
            all_entities[tmp_entity] = image_one['entity'][tmp_entity]
        else:
            all_entities[tmp_entity] += image_one['entity'][tmp_entity]
            
    for tmp_entity in image_two['entity']:
        if tmp_entity not in all_entities:
            all_entities[tmp_entity] = image_two['entity'][tmp_entity]
        else:
            all_entities[tmp_entity] += image_two['entity'][tmp_entity]
            
    sorted_entities = sorted(all_entities, key=lambda x: all_entities[x], reverse=True)
    sorted_entities = sorted_entities[:self.k] # select top k entities
    
    # embedding entities 
    tensorized_entities = list()
    tensorized_examples_for_one_frame = list()
    event_1_embeddings = list()
    event_2_embeddings = list()
    
    for tmp_entity in sorted_entities:
        tensorized_entities.append(self.word_embeddings[tmp_entity])
        
    tensorized_entities = torch.tensor(tensorized_entities).type(torch.float32).to(self.device)
    
    # split event 1 and event 2 and embedding
    for tail_dict in image_one['event'].values():
        for tmp_event_pair in tail_dict:
            event_1 = tmp_event_pair[0].split('$$')[0]
            event_2 = tmp_event_pair[0].split('$$')[1]
            
            for w in event_1.split(' '):
                event_1_embeddings.append(self.word_embeddings[w.lower()])
            
            for w in event_2.split(' '):
                event_2_embeddings.append(self.word_embeddings[w.lower()])
            
            if len(event_1_embeddings) > 1 and len(event_2_embeddings) > 1:
              tensorized_event_1 = torch.tensor(event_1_embeddings).type(torch.float32).to(self.device)
              tensorized_event_2 = torch.tensor(event_2_embeddings).type(torch.float32).to(self.device)
              
              bert_tokenized_event_1 = self.tokenizer.encode('[CLS] ' + event_1 + ' . [SEP]')
              bert_tokenized_event_2 = self.tokenizer.encode('[CLS] ' + event_2 + ' . [SEP]')

              tensorized_examples_for_one_frame.append({'event_1': tensorized_event_1,
                                                        'event_2': tensorized_event_2,
                                                        'bert_event_1': torch.tensor(bert_tokenized_event_1).to(self.device),
                                                        'bert_event_2': torch.tensor(bert_tokenized_event_2).to(self.device),
                                                        'entities': tensorized_entities,
                                                        'label': torch.tensor([tmp_event_pair[1]]).to(self.device),
                                                        'category': category,
                                                        'video_id': video_id,
                                                        'image_id': image_id,
                                                        'event_key': event_1}) # cause event String

    return tensorized_examples_for_one_frame
  
  def train_tensorize_frame(self, image_one, image_two, category, video_id, image_id):
  
    all_entities = dict()
    for tmp_entity in image_one['entity']:
        if tmp_entity not in all_entities:
            all_entities[tmp_entity] = image_one['entity'][tmp_entity]
        else:
            all_entities[tmp_entity] += image_one['entity'][tmp_entity]
    for tmp_entity in image_two['entity']:
        if tmp_entity not in all_entities:
            all_entities[tmp_entity] = image_two['entity'][tmp_entity]
        else:
            all_entities[tmp_entity] += image_two['entity'][tmp_entity]
    sorted_entities = sorted(all_entities, key=lambda x: all_entities[x], reverse=True)
    sorted_entities = sorted_entities[:self.k]
    
    tensorized_entities = list()
    
    for tmp_entity in sorted_entities:
        tensorized_entities.append(self.word_embeddings[tmp_entity])
    tensorized_entities = torch.tensor(tensorized_entities).type(torch.float32).to(self.device)
    
    tensorized_examples_for_one_frame = list()
    
    for tail_dict in image_one['event'].values():
        positive_list = list()
        negative_list = list()
        for tmp_event_pair in tail_dict:
            if tmp_event_pair[1] == 1:
                positive_list.append(tmp_event_pair)
            if tmp_event_pair[1] == 0:
                negative_list.append(tmp_event_pair)
        random.shuffle(negative_list)
        negative_list = negative_list[:len(positive_list)] ## why??1
        positive_list.extend(negative_list) # why ?? 2
        candidate_list = positive_list
        
        #train using positive candidate list
        for tmp_event_pair in candidate_list:
            event_1 = tmp_event_pair[0].split('$$')[0]
            event_2 = tmp_event_pair[0].split('$$')[1]
            event_1_embeddings = list()
            event_2_embeddings = list()
            for w in event_1.split(' '):
                event_1_embeddings.append(self.word_embeddings[w.lower()])
            for w in event_2.split(' '):
                event_2_embeddings.append(self.word_embeddings[w.lower()])
            if len(event_1_embeddings) > 1 and len(event_2_embeddings) > 1:
                tensorized_event_1 = torch.tensor(event_1_embeddings).type(torch.float32).to(self.device)
                tensorized_event_2 = torch.tensor(event_2_embeddings).type(torch.float32).to(self.device)
                
                bert_tokenized_event_1 = self.tokenizer.encode('[CLS] ' + event_1 + ' . [SEP]')
                bert_tokenized_event_2 = self.tokenizer.encode('[CLS] ' + event_2 + ' . [SEP]')
              
                tensorized_examples_for_one_frame.append({'event_1':tensorized_event_1,
                                                          'event_2':tensorized_event_2,
                                                          'bert_event_1': torch.tensor(bert_tokenized_event_1).to(self.device),
                                                          'bert_event_2': torch.tensor(bert_tokenized_event_2).to(self.device),
                                                          'entities': tensorized_entities,
                                                          'label': torch.tensor([tmp_event_pair[1]]).to(self.device),
                                                          'category': category,
                                                          'video_id': video_id,
                                                          'image_id': image_id,
                                                          'event_key': event_1
                                                          })
    
    
    return tensorized_examples_for_one_frame
