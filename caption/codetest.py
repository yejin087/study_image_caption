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

import os
import pickle
from collections import Counter
import nltk
from PIL import Image


def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image



folder = './data/test'
resized_folder = './data/test_resized/'

if not os.path.exists(resized_folder):
    os.makedirs(resized_folder)
    
image_files = os.listdir(folder)
num_images = len(image_files)

for i, image_file in enumerate(image_files):
    with open(os.path.join(folder, image_file), 'r+b') as f:
        with Image.open(f) as image:
            image = resize_image(image)
            image.save(os.path.join(resized_folder, image_file), image.format)

print("done resizing images...")