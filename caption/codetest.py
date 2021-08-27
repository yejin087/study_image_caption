"""
decode_lengths = 5
for t in range(max(decode_lengths)):
    batch_size_t = sum([l > t for l in decode_lengths])
print(batch_size_t)
# decode_lengths ( int) is not iterable


from pycocotools.coco import COCO
import matplotlib as plt
from collections import Counter
import nltk

annFile = './data/annotations/captions_train2014.json'
coco = COCO(annFile)
counter = Counter()
ids = coco.anns.keys()
for i, id in enumerate(ids):
    caption = str(coco.anns[id]['caption'])
    print(f'cption {caption} ')
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    print(f'token{tokens}')
    counter.update(tokens)
    


import pickle
import torch
import numpy as np
# Load vocabulary
with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

scores = pickle.load(open('scores.pkl','rb'))

hyp = []
decode_lengths = [22, 18, 14, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 9, 9, 9]  

PAD = 0
START = 1
END = 2
UNK = 3

_, preds = torch.max(scores, dim=2) # dim =  which dim to select max
preds = preds.tolist()
print(np.shape(preds), scores.shape)
temp_preds = list()

# pred에 있는 index중 pad ,start enc 제외하고 caption의 길이만클 다시 저장
for j, p in enumerate(preds): 
    #print(f' j is {j} p is {p} \n\n')
    pred = p[:9]
    pred = [w for w in pred if w not in [PAD, START, END]]
    temp_preds.append(pred)  # remove pads, start, and end

ind = temp_preds
hyp.extend(ind)
print(hyp)

hyp_sentence = []
for i in range(32):
    for word_idx in hyp[i]:
        hyp_sentence.append(vocab.idx2word[word_idx])

print('Hypotheses: '+" ".join(hyp_sentence))
 """
"""
Hypotheses: 
there is an elephant in the grass and there 
a herd of zebras drinking water from a small 
a sandwich sandwich with meat and meat on a 
a woman brushing her teeth with a blue and
man wearing wearing black outfit and black 
pants gear a large truck is in the snow next
 to the are ready ready ski the ski on on a 
 man that is holding a kite standing in this
 is not good good day with a of a kitchen that 
 has a counter with a oven black and white photograph
 of cars in front of a couple of parked cars sit parked 
 on some a snowboarder in a blue jacket a a buildings a 
 stop light that is next to some trees a man playing with 
 a ball on a tennis the meal on the plate is ready to be a man holding a bat outside of a batting a photo of a horse tied to to a people watching a elephant near some water and a a dining table with a table of donuts and three horses in a lake in a open area a busy road is being seen by a distance two cows are walking down a busy city street a table topped with plates and various of of a bird flies high in a clear sky . a skier is going downhill during a race . a man doing a jump jump on a rail a vase that has flowers flowers in it . a bunch of sheep are standing behind a fence the man is playing a a wii wii . a yellow train is at a train station two yellow yellow trucks trucks <unk> on road road
"""
""""
from nltk.translate.bleu_score import corpus_bleu
ref = ['two', 'plates', 'with', 'a', 'chicken', 'meal', 'and', 'vegetables',]
hypo = ['two', 'plates', 'with', 'a', 'chicken', 'meal', 'and', 'vegetables', '<end>', '<pad>', '<pad>', '<pad>', '<pad>', 
        '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>',
        '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>',
        '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']

ref1 = ['two', 'plates', 'with', 'a', 'chicken', 'meal', 'and', 'vegetables']
hypo1 = ['two', 'plates', 'with', 'a', 'chicken', 'meal', 'and', 'vegetables']

bleu_1_score = corpus_bleu([[ref]], [hypo], weights=(1, 0, 0, 0)) 

bleu_1_score1 = corpus_bleu([[ref1]], [hypo1], weights=(1, 0, 0, 0)) 
print('bleu , blue 1', bleu_1_score, bleu_1_score1)
# bleu , blue 1 = 0.1509433962264151 / 1.0
"""
"""   

import pickle
import processData
import json

vocab_path = './data/annotaions/uniq_vocab.pkl'
caption = pickle.load(open('./data/vocab.pkl','rb'))
causal = pickle.load(open('./Cause_Caption/data/annotations/causal_vocab.pkl','rb'))

caption_word_list = caption.idx2word.values() 
causal_word_list = causal.idx2word.values()

print(len(caption_word_list), len(causal_word_list))
caption_set = set()

for word in caption_word_list:
    caption_set.add(word)

print(len(caption_word_list))   
with open('../Image-Captions/unique_word.json', 'w') as outfile:
    json.dump(list(caption_set), outfile, indent=4)

for word in causal_word_list:
    caption_set.add(word)
   
counter = Counter()

for i, event in enumerate(unique_word_list): 
    caption = str(envent)
    #print('caption', caption)
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    #print('token', tokens)
    counter.update(tokens)

# ommit non-frequent words # token 단어 개수가 경계값 보다 많으면 저장한다 빈도수가 많지 않으면 생략.
words = [word for word, cnt in counter.items() if cnt >= threshold]

vocab = Vocabulary()
vocab.add_word('<pad>') # 0
vocab.add_word('<start>') # 1
vocab.add_word('<end>') # 2
vocab.add_word('<unk>') # 3

for i, word in enumerate(words):
    vocab.add_word(word)
    
 with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)
"""
import os
import nltk
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
import json
import pickle
json_path = './Cause_Caption/data/annotations/test.json'
# Load vocabulary
with open('./data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open(json_path, "r") as json_file:
    json = json.load(json_file)

length = len(json)

for i in range(length):
    caption = json[i]['event'] 
    print(caption)
    clean_caption = ' '.join([str(item) for item in caption])
    tokens = nltk.tokenize.word_tokenize(str(clean_caption).lower())
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    print(caption)

        
#img_id = coco.anns[ann_id]['image_id']
#path = coco.loadImgs(img_id)[0]['file_name']
