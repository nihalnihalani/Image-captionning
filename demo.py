''';==========================================
; Title:  demo.py
; Author: Chockalingam Ravi Sundaram 
; Last Updated: 11 Dec 2018
;==========================================
'''

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # used to convert tensors to cuda variables of gpu is available


# This function returns a caption for an image with a given beam size (This function is the same as eval.py file for a single image)
def eval_single_img(encoder, decoder, image_path, word_map, beam_size=3):
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
    image = transform(img)  

    # Encoder output for the image
    image = image.unsqueeze(0)  
    encoder_out = encoder(image)  
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)
 
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # Flatten encoding output as 1 x num_of_pixels x enc_dim
    num_pixels = encoder_out.size(1)

    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim) # Since the beam_size=k, we will treat the problem as having a batch size of k

    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)# Tenssor to store the top 'k' previous words at each step in the decoder (Now they are initialized with <start>)
    seqs = k_prev_words

    top_k_scores = torch.zeros(k, 1).to(device) # tensor to store the scores of the sequences (Now they are initialized with zeros)
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device) # tensor to store top k sequences (Now they are initialized with ones)

    # Decoding code starts
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # Embedding layer output

        att_weighted_enc, alpha = decoder.attention(encoder_out, h)  # Attention network output to get the weighed encoding and weights

        alpha = alpha.view(-1, enc_image_size, enc_image_size) 

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar in LSTM cell
        att_weighted_enc = gate * att_weighted_enc

        h, c = decoder.decode_step(torch.cat([embeddings, att_weighted_enc], dim=1), (h, c))# LSTM cell output with Attention network input concatenated with the Embedding layer output

        scores = decoder.fc(h)  
        scores = F.log_softmax(scores, dim=1)# Score = FC Layer + Softmax 
        scores = top_k_scores.expand_as(scores) + scores 

        if step == 1: # Only for the first step, all k points have the same score
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True) # Finding the top k scores and their respective indices

        prev_word_inds = top_k_words / vocab_size  
        next_word_inds = top_k_words % vocab_size # Getting the index of the next word in the Word map(or vocab)

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1) # Adding new words to the sequences 
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']] # List of sequences which are incomplete (i.e did not reach <end>)
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        if k == 0: # Stop when you are done generating sequences
            break
     
        # Proceed with incomplete sequences
        seqs = seqs[incomplete_inds]
        # Getting the previous states of the LSTM networks at the incomplete indices
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > 50: # Stop if things are going on for a long time!!!
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores)) # Choosing the sequence with the highest score
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas

# This function returns a plot with captions and weighed images for every word
def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq] # transforms indices to words

    for t in range(len(words)):
        if t > 50: # Stop when the number of words in the caption is greater than 50.
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t]
        if smooth:
            alpha = skimage.transform.pyramid_expand(np.asarray(current_alpha), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(np.asarray(current_alpha), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    #Loading the trained model for caption generation and visualization
    print('Loading checkpoint!!!')
    checkpoint = torch.load(args.model,map_location='cpu') # remove map_location when using gpu
    decoder = checkpoint['decoder']
    decoder = decoder.to(device) # Converts the decoder model to CUDA variables if gpu is available
    decoder.eval() # No back-prop used as the decoder is only used for evaluation purposes
    encoder = checkpoint['encoder']
    encoder = encoder.to(device) # Converts the encoder model to CUDA variables if gpu is available
    encoder.eval() # No back-prop used as the encoder is only used for evaluation purposes

    # Load word map
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  

    # Calling the function to get captions and weights
    seq, alphas = eval_single_img(encoder, decoder, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Using the captions and weights from eval_single_img() call to generate subplots of images with captions
    visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
