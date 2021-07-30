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

from opts import parser
from DataLoader import *
from Model import *

def train(modelname,model, data, learning_rate):

  loss_func = torch.nn.CrossEntropyLoss()
  test_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  
  print( '##start traning##')
  all_loss = 0
  random.shuffle(data)
  train_loss_list = []
  model.train()
 
  for tmp_example in tqdm(data):
    if modelname == 'ResNetAsContext':
        final_prediction = model(event_1=tmp_example['bert_event_1'], event_2=tmp_example['bert_event_2'],
                    resnet_representation=tmp_example['resnet_representation'])
    else:
        final_prediction = model(event_1=tmp_example['bert_event_1'], event_2=tmp_example['bert_event_2'],
                    entities=tmp_example['entities'])     
    loss = loss_func(final_prediction, tmp_example['label'])
    test_optimizer.zero_grad()
    loss.backward()
    test_optimizer.step()
    
    train_loss_list.append(loss.item())
    all_loss += loss.item()

  return  all_loss / len(data)

def eval(modelname, model, data):

  recall_list = [1, 2, 3, 5, 10]
  model.eval()
  random.shuffle(data)
  prediction_dict = dict()
  
  for video in range(100):
      prediction_dict['video_'+str(video)] = dict()
      for image in range(4):
          prediction_dict['video_'+str(video)]['image_'+str(image)] = dict()
  gt_positive_example = 0
  pred_positive_example = dict()
  
  for top_k in recall_list:
      pred_positive_example['top'+str(top_k)] = 0
  
  
  for tmp_example in data:
    if modelname == 'ResNetAsContext':
            final_prediction = model(event_1=tmp_example['bert_event_1'], event_2=tmp_example['bert_event_2'],
                        resnet_representation=tmp_example['resnet_representation'])
    else:
        final_prediction = model(event_1=tmp_example['bert_event_1'], event_2=tmp_example['bert_event_2'],
                    entities=tmp_example['entities'])

    softmax_prediction = F.softmax(final_prediction, dim = 1)

    tmp_one_result = dict()
    tmp_one_result['True_score'] = softmax_prediction.data[0][1]
    tmp_one_result['label'] = tmp_example['label'].item()

    if tmp_example['event_key'] not in prediction_dict['video_'+str(tmp_example['video_id'])]['image_'+str(tmp_example['image_id'])].keys():
        prediction_dict['video_'+str(tmp_example['video_id'])]['image_'+str(tmp_example['image_id'])][tmp_example['event_key']] = list()
        
    prediction_dict['video_' + str(tmp_example['video_id'])]['image_' + str(tmp_example['image_id'])][tmp_example['event_key']].append(tmp_one_result)

    if tmp_example['label'].data[0] == 1:
        gt_positive_example += 1
  
  for video in range(100):
      for image in range(4):
          current_predict = prediction_dict['video_'+str(video)]['image_'+str(image)]
          for key in current_predict:
              current_predict[key] = sorted(current_predict[key], key=lambda x: (x.get('True_score', 0)), reverse=True)
              for top_k in recall_list:
                  tmp_top_predict = current_predict[key][:top_k]
                  for tmp_example in tmp_top_predict:
                      if tmp_example['label'] == 1:
                          pred_positive_example['top' + str(top_k)] += 1
  
  recall_result = dict()
  for top_k in recall_list:
      recall_result['Recall_' + str(top_k)] = pred_positive_example['top' + str(top_k)] / gt_positive_example
  
  return recall_result


def main():  
  
    args = parser.parse_args()

    # Use gpu
    logging.basicConfig(level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Current device:', device)

    n_gpu = torch.cuda.device_count()
    print('Number of gpu:', n_gpu)

    torch.cuda.get_device_name(0)
    modelname = args.model 
    learning_rate = args.lr    
    epochs = args.epoch
    vaildation = args.val
    
    if modelname == 'Full-Model':
      tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      current_model = BERTCausal.from_pretrained('bert-base-uncased')
    elif modelname == 'NoContext':
        current_model = NoContext.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif modelname == 'NoAttention':
        current_model = NoAttention.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif modelname == 'ResNetAsContext':
        current_model = ResNetAsContext.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif modelname == 'GPT-2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        current_model = GPTCausal.from_pretrained('gpt2')
    else:
      print('Please Choose one model!!')

    print('Using Model :', args.model)

    current_model.to(device)
    all_data = DataLoader(tokenizer,device)
    
    best_dev_performance = dict()
    final_performance = dict()

    for top_k in [1, 2, 3, 5, 10]:
        best_dev_performance['Recall_'+str(top_k)] = 0

    # train
    train_data = all_data.load_train()  
    dev_data = all_data.load_dev()

    for i in range(epochs):
        print('##Iteration:', i+1, '|')
        train_loss_avg = train(modelname, current_model, train_data, learning_rate)
        print(f'[{i+1}/ train loss] : {train_loss_avg} \n')

    print('\nfinish train.')

    #vaildation

    dev_performance = eval(modelname, current_model, dev_data)
    print('Dev accuracy:', dev_performance)

    for top_k in [1, 2, 3, 5, 10]:
        if dev_performance['Recall_' + str(top_k)] > best_dev_performance['Recall_' + str(top_k)]:
            best_dev_performance['Recall_' + str(top_k)] = dev_performance['Recall_' + str(top_k)]                
            torch.save(current_model.state_dict(), 'models/' + 'Recall' + str(top_k) + '_' + args.model + '.pth')
            
    print('\nfinish evalutation.')

  
if __name__ == '__main__' :
  main() 