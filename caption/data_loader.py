'''
This code is mainly taken from the following github repositories:
1.  parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This script loads the COCO dataset in batches to be used for training/testing
''' 

import os
import nltk
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms

class DataLoader(data.Dataset):
    def __init__(self, root, json, vocab, transform=None):

        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)# image에 해당하는 caption이  index로 바뀐 tensor
        return image, target

    def __len__(self):
        return len(self.ids)

def collate_fn(data): # sample을 batch로 합친다.
    data.sort(key=lambda  x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions] # batch 에 포함된 각 캡션의 단어 길이 list
    targets = torch.zeros(len(captions), max(lengths)).long() # (캡션길이,가장 긴 caption 길이) 행렬 tensor 생성
    for i, cap in enumerate(captions):
        #print('captions, cp, i', len(captions), i, cap)
        end = lengths[i] # i 째 caption의 단어길이
        targets[i, :end] = cap[:end] # targets의 i번째 caption 저장????  
    return images, targets, lengths

def get_loader(method, vocab, batch_size):

    # train/validation paths
    if method == 'train':
        root = 'data/train2014_resized'
        json = 'data/annotations/captions_train2014.json'
    elif method =='val':
        root = 'data/val2014_resized'
        json = 'data/annotations/captions_val2014.json'
    elif method == 'test':
        root = 'data/test_resized'

    # rasnet transformation/normalization
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])

    coco = DataLoader(root=root, json=json, vocab=vocab, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=1,
                                              collate_fn=collate_fn)
    return data_loader