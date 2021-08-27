'''
Parts of this code were incorporated from the following github repositories:
1. parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This script has the Encoder and Decoder models and training/validation scripts. 
Edit the parameters sections of this file to specify which models to load/run
''' 

# -*- coding: utf-8 -*-

import pickle
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from data_loader import get_loader
from nltk.translate.bleu_score import corpus_bleu
from processData import Vocabulary
from tqdm import tqdm
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import skimage.transform
from scipy.misc import imread, imresize
from PIL import Image
import matplotlib.image as mpimg
from torchtext.vocab import Vectors, GloVe
from scipy import misc
from pytorch_pretrained_bert import BertTokenizer, BertModel
import imageio
import pandas as pd
from pandas import DataFrame
import time

###################
# START Parameters
###################

# hyperparams
grad_clip = 5.
num_epochs = 20
batch_size = 32 
decoder_lr = 0.0004

# if both are false them model = baseline

from_checkpoint = False
train_model = True
valid_model = False

###################
# END Parameters
###################

# loss
class loss_obj(object):
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#####################
# Encoder RASNET CNN
#####################
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, images):
        out = self.adaptive_pool(self.resnet(images))
        # batch_size, img size, imgs size, 2048
        out = out.permute(0, 2, 3, 1)
        return out

####################
# Attention Decoder
####################
class Decoder(nn.Module):

    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.encoder_dim = 2048
        self.attention_dim = 512

        # word embedding dimension
        self.embed_dim = 512

        self.decoder_dim = 512
        self.vocab_size = vocab_size
        self.dropout = 0.5
        
        # soft attention
        self.enc_att = nn.Linear(2048, 512) # linear layer to transform 
        self.dec_att = nn.Linear(512, 512)
        self.att = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # decoder layers
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)
        self.h_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.c_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        # init variables
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
    
        # always fine-tune embeddings (even with GloVe)
        for p in self.embedding.parameters():
            p.requires_grad = True

    def forward(self, encoder_out, encoded_captions, caption_lengths):    
    
        batch_size = encoder_out.size(0) # 32
        encoder_dim = encoder_out.size(-1) #2048
        vocab_size = self.vocab_size # 8853 -> 8969
        
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) # view 텐서의 크기를 ( 32,?, 2048) 로 변경
        #print(f'encoder out {encoder_out.shape}') (32,196,2048)
        num_pixels = encoder_out.size(1)
        #print(f'num_pixels {num_pixels}') 196
       
        sort_ind = sorted( range(len(caption_lengths)), key=lambda k: caption_lengths[k], reverse=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        dec_len = [x-1 for x in caption_lengths] # caption end token 전까지 길이
        max_dec_len = max(dec_len) # caption len 중 가장 긴 caption 길이
        
        # load regular embeddings
        embeddings = self.embedding(encoded_captions)
        # print(f' embedding shaep {embeddings.shape}') torch.Size([32,20/21/19/2,768]) , embedding size = 768
            
        # init hidden state
        avg_enc_out = encoder_out.mean(dim=1)
        h = self.h_lin(avg_enc_out)
        c = self.c_lin(avg_enc_out)
        # print(f' h shape {h.shape} , c shape {c.shape}') h, c 모두 ([32,512])

        predictions = torch.zeros(batch_size, max_dec_len, vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_dec_len, num_pixels).to(device)

        for t in range(max(dec_len)):
            batch_size_t = sum([l > t for l in dec_len ])
            #print(f'batch_size_t {batch_size_t} t {t} l {l > t for l in dec_len } dec_len {dec_len} ')
            # soft-attention
            enc_att = self.enc_att(encoder_out[:batch_size_t])
            dec_att = self.dec_att(h[:batch_size_t])
            att = self.att(self.relu(enc_att + dec_att.unsqueeze(1))).squeeze(2)
            alpha = self.softmax(att)
            attention_weighted_encoding = (encoder_out[:batch_size_t] * alpha.unsqueeze(2)).sum(dim=1)
        
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            batch_embeds = embeddings[:batch_size_t, t, :]            
            cat_val = torch.cat([batch_embeds.double(), attention_weighted_encoding.double()], dim=1)
            
            h, c = self.decode_step(cat_val.float(),(h[:batch_size_t].float(), c[:batch_size_t].float()))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds # preds shape is (Batch size_t, vocab_size)
            #print('preds',preds.shape,predictions.shape )
            alphas[:batch_size_t, t, :] = alpha
            
        # preds, sorted capts, dec lens, attention wieghts
        return predictions, encoded_captions, dec_len, alphas

# vocab indices
PAD = 0
START = 1
END = 2
UNK = 3 # unknown words not exist in vocabulary

# Load vocabulary
with open('./Cause_Caption/data/annotations/unique_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# load data
train_loader = get_loader('train', vocab, batch_size)
val_loader = get_loader('val', vocab, batch_size)

#############
# Init model
#############

criterion = nn.CrossEntropyLoss().to(device)

if from_checkpoint:

    encoder = Encoder().to(device)
    decoder = Decoder(vocab_size=len(vocab),use_glove=glove_model, use_bert=bert_model).to(device)

    if torch.cuda.is_available():
        print('Pre-Trained Baseline Model')
        encoder_checkpoint = torch.load('./checkpoints/causal/base/encoder_epoch6')
        decoder_checkpoint = torch.load('./checkpoints/causal/base/decoder_epoch6')

    encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr)
    decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
    decoder_optimizer.load_state_dict(decoder_checkpoint['optimizer_state_dict'])
    
else:
    encoder = Encoder().to(device)
    decoder = Decoder(vocab_size=len(vocab)).to(device)
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr)

###############
# Train model
###############

def train():
    print("Started training...")
    for epoch in tqdm(range(num_epochs)):
        decoder.train() 
        encoder.train()

        losses = loss_obj()
        num_batches = len(train_loader) ## 배치 사이즈 32

        for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader)):

            imgs = encoder(imgs.to(device))
            caps = caps.to(device)

            scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]

            targets = caps_sorted[:, 1:]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            loss = criterion(scores, targets).to(device)

            loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

            decoder_optimizer.zero_grad()
            loss.backward()

            # grad_clip decoder
            for group in decoder_optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

            decoder_optimizer.step()

            losses.update(loss.item(), sum(decode_lengths))

        if (epoch)%5==0:    
            torch.save({
                'epoch': (epoch),
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': decoder_optimizer.state_dict(),
                'loss': loss,
                }, './checkpoints/base_sort/decoder_epoch'+str(epoch))

            torch.save({
                'epoch': (epoch),
                'model_state_dict': encoder.state_dict(),
                'loss': loss,
                }, './checkpoints/base_sort/encoder_epoch'+str(epoch))
            print('epoch '+str(epoch)+'/', num_epochs,'Batch '+str(i)+'/'+str(num_batches)+' loss:'+str(losses.avg))
                
            print('epoch checkpoint saved')

    print("Completed training...")  

#################
# Validate model
#################

def print_sample(hypotheses, references, test_references,imgs, alphas,losses):
    
    data = dict()
    
    for count , (reference,hypothesis, test_reference) in enumerate(zip(references, hypotheses, test_references)):
        #bleu_hypo_ref = []
        #bleu_1_score = corpus_bleu([[test_reference]], [hypothesis], weights=(0, 0, 0, 1)) # weight : value of n ( n-gram)
        #bleu_hypo_ref.append(bleu_1_score)
        
        #print(test_reference)
        hyp_sentence = []
        for word_idx in hypothesis: 
            hyp_sentence.append(vocab.idx2word[word_idx])
        
        ref_sentence = []
        for test in test_reference:
            for word in test:
                ref_sentence.append(vocab.idx2word[word])
        
        print('hypo',' '.join(hyp_sentence), '\nref', ' '.join(ref_sentence))

    bleu_1 = corpus_bleu(test_references, hypotheses, weights=(1, 0, 0, 0))
    #bleu_4 = corpus_bleu([[test_references]], hypotheses, weights=(0, 0, 0, 1))
    print("BLEU-1: "+str(bleu_1))
    #print("BLEU-1: "+str(bleu_1))

def visualize(imgs, img_seq_num, alphas, losses, hyp_sentence, smooth=False):

    img_dim = 336 # 14*24

    
    img = imgs[0][img_seq_num] 
    imageio.imwrite(f'./visualize/img{img_seq_num}.jpg', img)


    image = Image.open(f'./visualize/img{img_seq_num}.jpg')
    image = image.resize([img_dim, img_dim], Image.LANCZOS)
    
    for t in range(len(hyp_sentence)):
        
        if t > 50:
            break
            
        plt.subplot(np.ceil(len(hyp_sentence) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (hyp_sentence[t]), color='black', backgroundcolor='white', fontsize=7)
        plt.imshow(image)
        print('####t',t)
        current_alpha = alphas[img_seq_num][t, :]
        
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.cpu().numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.cpu().numpy(), [14 * 24, 14 * 24])
        
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    plt.savefig(f'./visualize/img{img_seq_num}.jpg')

    
def validate():

    references = [] 
    test_references = []
    hypotheses = [] 
    all_imgs = []
    all_alphas = []

    print("Started validation...")
    decoder.eval()
    encoder.eval()

    losses = loss_obj()

    num_batches = len(val_loader)
    
    #validation start
    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(tqdm(val_loader)):

        imgs_jpg = imgs.numpy() 
        imgs_jpg = np.swapaxes(np.swapaxes(imgs_jpg, 1, 3), 1, 2)
        
        # Forward prop.
        imgs = encoder(imgs.to(device))
        caps = caps.to(device)

        scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        loss = criterion(scores_packed, targets_packed)
        loss += ((1. - alphas.sum(dim=1)) ** 2).mean()
        losses.update(loss.item(), sum(decode_lengths))

         # References
        for j in range(targets.shape[0]):
            img_caps = targets[j].tolist() # validation dataset only has 1 unique caption per img
            clean_cap = [w for w in img_caps if w not in [PAD, START, END]]  # remove pad, start, and end
            img_captions = list(map(lambda c: clean_cap,img_caps))
            test_references.append([clean_cap])
            references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            pred = p[:decode_lengths[j]]
            pred = [w for w in pred if w not in [PAD, START, END]]
            temp_preds.append(pred)  # remove pads, start, and end
        preds = temp_preds
        hypotheses.extend(preds)
        
        if i == 0:
            all_alphas.append(alphas)
            all_imgs.append(imgs_jpg)

    # end validation
    end = time.time()

    print("Completed validation... time elapsed" , end - start)
    
    print_sample(hypotheses, references, test_references, all_imgs, all_alphas,losses)
    #print("Completed validation... hypo shape" ,all_imgs.size)
     
######################
# Run training/validation
######################

if train_model:
    train()

if valid_model:
    validate()
