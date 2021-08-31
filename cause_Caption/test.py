from processData import Vocabulary
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image
from main2 import Decoder,Encoder
import pickle
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                 
    
    transform = transforms.Compose([normalize])
    #print(img.shape)
    image = transform(img)# (3, 256, 256)
    print(image.shape)
    

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    print(enc_image_size)
    encoder_dim = encoder_out.size(3)
    print(encoder_dim)
    
    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    avg_enc_out = encoder_out.mean(dim=1)
    h = decoder.h_lin(avg_enc_out)
    c = decoder.c_lin(avg_enc_out)
    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        #awe, alpha = decoder.att(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        
        enc_att = decoder.enc_att(encoder_out)
        dec_att = decoder.dec_att(h)
        att = decoder.att(decoder.relu(enc_att + dec_att.unsqueeze(1))).squeeze(2)
        alpha = decoder.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        
        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        attention_weighted_encoding = gate * attention_weighted_encoding

        h, c = decoder.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            #print('$$',top_k_scores,'\n@@@@\n', top_k_words,scores.shape)
            
                
        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)
        #print('@@@',prev_word_inds,next_word_inds)
        
        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        #print('##',seqs)
        
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
            
        print(f'step {step}')
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas



def visualize_att(image_path, seq, alphas, rev_word_map, smooth=False):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24],Image.ANTIALIAS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=10)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
            
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
            
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    
    plt.show()
    #plt.savefig(f'./visualize/test/{os.path.basename(image_path)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--beam_size','-bs',type=int, help ='beam search size')
    #parser.add_argument('--gpu_number','-gn',help='select visible device')
    args = parser.parse_args()
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    vocab = pickle.load(open('./data/annotations/unique_vocab.pkl', 'rb'))

    word_map = vocab.word2idx
    rev_word_map = vocab.idx2word
    
    encoder = Encoder().to(device)
    decoder = Decoder(vocab_size=len(vocab)).to(device)

    print('load Pre-Trained Baseline Model')
    encoder_checkpoint = torch.load('../checkpoints/causal2/encoder_epoch100')
    decoder_checkpoint = torch.load('../checkpoints/causal2/decoder_epoch100')
 
    encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
    print('sucessgully loaded Pre-Trained Model')
    folder = './data/images/testing'
    image_files = os.listdir(folder)
    num_images = len(image_files)
    
    for i, image_file in enumerate(image_files[:10]):
        img_path = os.path.join(folder, image_file)
                
        # Encode, decode with attention and beam search
        seq, alphas = caption_image_beam_search(encoder, decoder, img_path, word_map, args.beam_size)
        alphas = torch.FloatTensor(alphas)

        # Visualize caption and attention of best sequence
        visualize_att(img_path, seq, alphas, rev_word_map)
    