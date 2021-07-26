import argparse
from opts import parser
from DataLoader import *
from Model import *

def test(model, data):

  recall_list = [1, 2, 3, 5, 10]
  model.eval()
  random.shuffle(data)
  
  #initialize
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
      """
      if args.model == 'ResNetAsContext':
          final_prediction = model(event_1=tmp_example['bert_event_1'], event_2=tmp_example['bert_event_2'],
                      resnet_representation=tmp_example['resnet_representation'])
      else:
     """
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
          #print('truth example {} \n'.format(tmp_example))
          gt_positive_example += 1
  
  for video in range(100):
      for image in range(4):
          current_predict = prediction_dict['video_'+str(video)]['image_'+str(image)]
          for key in current_predict:
              current_predict[key] = sorted(current_predict[key], key=lambda x: (x.get('True_score', 0)), reverse=True)
              #print('current preditct key : {}'.format(current_predict[key]))
              for top_k in recall_list:
                  tmp_top_predict = current_predict[key][:top_k]
                  for tmp_example in tmp_top_predict:
                      if tmp_example['label'] == 1:
                          #print('pred example {} \n'.format(tmp_example))
                          pred_positive_example['top' + str(top_k)] += 1
  
  recall_result = dict()
  for top_k in recall_list:
      recall_result['Recall_' + str(top_k)] = pred_positive_example['top' + str(top_k)] / gt_positive_example
  
  # return correct_count / len(data)
  return recall_result
  
  
def test_by_type(model, data, recall_k):
  correct_count = dict()
  all_count = dict()
  correct_count['overall'] = 0
  all_count['overall'] = 0
  model.eval()d
  random.shuffle(data)

  # initialize
  prediction_dict = dict()
  for video in range(100):
      prediction_dict['video_' + str(video)] = dict()
      for image in range(4):
          prediction_dict['video_' + str(video)]['image_' + str(image)] = dict()

  for tmp_example in data:
      if args.model == 'ResNetAsContext':
          final_prediction = model(event_1=tmp_example['bert_event_1'], event_2=tmp_example['bert_event_2'],
                      resnet_representation=tmp_example['resnet_representation'])
      else:
          final_prediction = model(event_1=tmp_example['bert_event_1'], event_2=tmp_example['bert_event_2'],
                      entities=tmp_example['entities'])


      softmax_prediction = F.softmax(final_prediction, dim=1)

      if tmp_example['category'] not in correct_count:
          correct_count[tmp_example['category']] = 0
      if tmp_example['category'] not in all_count:
          all_count[tmp_example['category']] = 0

      tmp_one_result = dict()
      tmp_one_result['True_score'] = softmax_prediction.data[0][1]
      tmp_one_result['label'] = tmp_example['label'].item()
      tmp_one_result['category'] = tmp_example['category']

      if tmp_example['event_key'] not in prediction_dict['video_' + str(tmp_example['video_id'])][
          'image_' + str(tmp_example['image_id'])].keys():
          prediction_dict['video_' + str(tmp_example['video_id'])]['image_' + str(tmp_example['image_id'])][
              tmp_example['event_key']] = list()
      prediction_dict['video_' + str(tmp_example['video_id'])]['image_' + str(tmp_example['image_id'])][
          tmp_example['event_key']].append(tmp_one_result)

      if tmp_example['label'].data[0] == 1:
          all_count['overall'] += 1
          all_count[tmp_example['category']] += 1

  for video in range(100):
      for image in range(4):
          current_predict = prediction_dict['video_' + str(video)]['image_' + str(image)]
          for key in current_predict:
              current_predict[key] = sorted(current_predict[key], key=lambda x: (x.get('True_score', 0)),
                                            reverse=True)
              # print(current_predict[key])
              tmp_top_predict = current_predict[key][:recall_k]
              for tmp_example in tmp_top_predict:
                  if tmp_example['label'] == 1:
                      correct_count[tmp_example['category']] += 1
                      correct_count['overall'] += 1

  accuracy_by_type = dict()
  for tmp_category in all_count:
      accuracy_by_type[tmp_category] = correct_count[tmp_category] / all_count[tmp_category]

  return accuracy_by_type

def main():
  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Current device:', device)  
  
  all_data = DataLoader(device,args.k)
  test_data = all_data.load_test()
  
  if args.model == 'Full-Model':
      current_model = BERTCausal.from_pretrained('bert-base-uncased')
  elif args.model == 'NoContext':
      current_model = NoContext.from_pretrained('bert-base-uncased')
  elif args.model == 'NoAttention':
      current_model = NoAttention.from_pretrained('bert-base-uncased')
  elif args.model == 'ResNetAsContext':
      current_model = ResNetAsContext.from_pretrained('bert-base-uncased')
  elif args.model == 'GPT-2':
      #tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
      current_model = GPTCausal.from_pretrained('gpt2')
  else:
      print('Please Choose one model!!')
  
  print('Using Model :', args.model)
    
  current_model.to(device)
  
  for top_k in [1, 2, 3, 5, 10]:
    current_model.load_state_dict(torch.load('models/' + 'Recall' + str(top_k) + '_' + args.model + '.pth'))
    test_performance = test(current_model, test_data)
    
  print('Test accuracy:', test_performance)
  
if __name__ == '__main__' :
  main()
  