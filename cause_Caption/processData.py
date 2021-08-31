'''
This code is mainly taken from the following github repositories:
1.  parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This script processes the COCO dataset
'''  

import os
import pickle
from collections import Counter
import nltk
from PIL import Image
from pycocotools.coco import COCO
import json

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json_file_path, threshold):
    
    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)
        
    counter = Counter()

    length = len(json_data)   
    for i in range(length):
        caption = json_data[i]['event'] 
        clean_caption = ' '.join([str(item) for item in caption])
        tokens = nltk.tokenize.word_tokenize(clean_caption.lower())
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
        print('##',i,word)
        
    return vocab

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
    image = image.resize([256, 265], Image.ANTIALIAS)
    return image

def main(caption_path,vocab_path,threshold):
    vocab = build_vocab(caption_path,threshold)

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    print('finish build vocab')
"""
    print("resizing images...")
    #splits = ['val','train']

    #for split in splits:
    folder = './data/causal/training'
    resized_folder = './data/causal/training_resized' 
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
        
    image_files = os.listdir(folder)
    num_images = len(image_files)
    for i, image_file in enumerate(image_files):
        with open(os.path.join(folder, image_file), 'r+b') as f:
            with Image.open(f).convert("RGB") as image:
                image = resize_image(image)
                image.save(os.path.join(resized_folder, image_file), image.format)

    print("done resizing images...")
"""    

caption_path = '/home/muser/Context/Image-Captions/Cause_Caption/data/annotations/test.json'
vocab_path = './data/annotations/train_word_vocab.pkl'
threshold = 0

#main(caption_path,vocab_path,threshold)


"""
import pickle

vocab_path = './data/annotaions/uniq_vocab.pkl'
caption = pickle.load(open('./data/annotations/vocab.pkl','rb'))
uni = pickle.load(open('./data/annotations/unique_vocab.pkl','rb'))

caption_word_list = caption.idx2word.values() 
uni_word_list = uni.idx2word.values()

print(len(caption_word_list), len(uni_word_list))

caption_set = set()
for word in caption_word_list:
    caption_set.add(word)

uni_set = set()
for word in uni_word_list:
    uni_set.add(word)
    
diff = caption_set.difference(uni_set)
print( diff )
"""