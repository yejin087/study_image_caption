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

def integrate_vocab():
    vocab1_path = '/home/muser/Context/Image-Captions/Cause_Caption/data/annotations/causal_vocab.pkl'
    vocab2_path = '/home/muser/Context/Image-Captions/Cause_Caption/data/annotations/vocab.pkl'
    
    v1 = pickle.load(open(vocab1_path,'rb'))
    v2 = pickle.load(open(vocab2_path,'rb'))

    v1_word_list = v1.idx2word.values() 
    v2_word_list = v2.idx2word.values()

    print(len(v1_word_list), len(v2_word_list))

    v1_set = set()
    for word in v1_word_list:
        v1_set.add(word)

    v2_set = set()
    for word in v2_word_list:
        v2_set.add(word)
     
    diff = v2_set.difference(v1_set)
    print(len(diff))

    integrate = v1_set.union(v2_set)
    print(len(integrate))
    
    vocab = Vocabulary()
    vocab.add_word('<pad>') # 0
    vocab.add_word('<start>') # 1
    vocab.add_word('<end>') # 2
    vocab.add_word('<unk>') # 3
    
    for i, word in enumerate(integrate):
        vocab.add_word(word)
        #print('##',i,word)
    
    print(len(integrate))
    
    with open('/home/muser/Context/Image-Captions/Cause_Caption/data/annotations/unique_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

def diff_vocab():
    vocab1_path = '/home/muser/Context/Image-Captions/data/annotations/train_word_vocab.pkl'
    vocab2_path = '/home/muser/Context/Image-Captions/Cause_Caption/data/annotations/unique_vocab.pkl'
    
    v1 = pickle.load(open(vocab1_path,'rb'))
    v2 = pickle.load(open(vocab2_path,'rb'))

    v1_word_list = v1.idx2word.values() 
    v2_word_list = v2.idx2word.values()

    print(len(v1_word_list), len(v2_word_list))

    v1_set = set()
    for word in v1_word_list:
        v1_set.add(word)

    v2_set = set()
    for word in v2_word_list:
        v2_set.add(word)
     
    diff = v1_set.difference(v2_set)
    print(len(diff), diff)

def main(caption_path,vocab_path,threshold):
    #vocab = build_vocab(json_file_path=caption_path,threshold=threshold)
    #with open(vocab_path, 'wb') as f:
    #    pickle.dump(vocab, f)
    #print("finish vocab...")
    integrate_vocab()
    #diff_vocab()
    
caption_path = '/home/muser/Context/Image-Captions/Cause_Caption/data/annotations/train.json'
vocab_path = '/home/muser/Context/Image-Captions/Cause_Caption/data/annotations/train_word_vocab.pkl'
threshold = 0

main(caption_path,vocab_path,threshold)
