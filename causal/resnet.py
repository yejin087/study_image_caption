
# -*- coding: utf-8 -*

import os, torch, glob
import numpy as np
from torch.autograd import Variable
from PIL import Image 
from torchvision import models, transforms
import torch.nn as nn
import shutil
import pandas as pd
import pickle

data_dir = './data/images/vaildation'
features_dir = './feature'

def extractor(img_path, net, use_gpu):

    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    img = Image.open(img_path)
    img = transform(img)

    t_img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    
    my_embedding = torch.zeros(2048)
    
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
        
    h = layer.register_forward_hook(copy_data)
    
    model(t_img)
    h.remove()
    
    return my_embedding

if __name__ == '__main__':
    
    # get image file names list
    abs_list = []
    filenames = os.listdir(data_dir)

    for filename in filenames:
        full_filename = os.path.join(os.getcwd(),data_dir[2:], filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == '.jpg': 
            #print(full_filename)
            abs_list.append(full_filename)
    
    # select resnet layer you want to extract feature values
    model = models.resnet152(pretrained=True) 
    layer = model._modules.get('avgpool')
    model.eval()
    
    #for param in convnet.parameters():
    #    param.requires_grad = False 
     
    use_gpu = torch.cuda.is_available()

    featuer_dict = dict()
    # set the each file name to extract function 
    # save the image name and feature pair in dataframe 
    for x_path in abs_list:
        file_name = x_path.split('/')[-1]
        #fx_path = os.path.join(features_dir, file_name + '.txt')
        feature_val = extractor(x_path, model, use_gpu)
        featuer_dict[file_name] = feature_val.cpu().numpy()
        print(featuer_dict)
        
    #df = pd.DataFrame(list(featuer_dict.items()), columns = ['imageFileName' , 'extractedFeatures'])
    #df.to_pickle('./feature/test_feature.pkl')   
    # Store data (serialize)
    with open('./feature/val_feature.pkl', 'wb') as handle:
      pickle.dump(featuer_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('done extracting')
