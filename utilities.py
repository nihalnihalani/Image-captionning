''';==========================================
; Title:  utilities.py 
; Author: Ashkan Kazemi
; Last Updated:   01 Dec 2018
;==========================================
'''
import json
import os
from collections import Counter
from random import choice, sample, seed

import h5py
import numpy as np
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm

# creating files for train, val, test
# dataset should be either of'coco', 'flickr8k', 'flickr30k'
# images_split_caption_json_path: json file path for images split and captions corresponding to it. Here we are using Karpathy image split.
# captions_per_image : number of captions per image
# min_wor_freq:  lower word threshold. less frequent words are made as <unk>s
# max_len: upper threshold for number of words per image
# output_folder: output folder to save created files

def createInput(dataset, images_split_caption_json_path, image_folder, captions_per_image, min_word_count, output_folder,
                       max_len=100):
    if not dataset in {'coco', 'flickr8k', 'flickr30k'}:
        raise AssertionError("wrong dataset input")

    # load data from images_split_caption  JSON for corresponding dataset
    file = open (images_split_caption_json_path)
    data = json.load(file)
    file.close()

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_count = Counter()

    for img in data['images']:
        captions = []
        for caps in img['sentences']:
            # Update word count and append captions
            word_count.update(caps['tokens'])
            if len(caps['tokens']) <= max_len:
                captions.append(caps['tokens'])
        #if caption length is 0 skip this image
        if len(captions) == 0:
            continue

        # generate the path for output files

        if dataset == 'coco':
            path = os.path.join(image_folder, img['filepath'], img['filename'])
        else:
            os.path.join(image_folder, img['filename'])

        # fill the train_image_paths, val_image_paths, test_image_paths and corresponding captions as per the split mentioned in the json 
        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # check if the length of the image_paths array is equal to image_captions
    assert (len(train_image_paths) == len(train_image_captions))," train_image_path assertion error"
    assert (len(val_image_paths) == len(val_image_captions))," val_image_path assertion error"
    assert (len(test_image_paths) == len(test_image_captions))," test_image_path assertion error"

    # Create word map
    words = [w for w in word_count.keys() if word_count[w] > min_word_count]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_count) + '_min_word_count'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, caps in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in caps] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(caps))

                    # Find caption lengths
                    c_len = len(caps) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)



# An embedding tensor is created for the given word map given file containing embeddings (Glove Format)
def load_embeddings(emb_file, word_map):
    
    # Find embedding dimension
    file = open (emb_file)
    dims = len(file.readline().split(' ')) - 1
    file.close()
    
    # Create tensor to hold embeddings
    embeddings = torch.FloatTensor(len(set(word_map.keys())), dims)

    # initialize embedding tensor with values from the uniform distribution.
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)

    # Read emb_file
    print("\nReading emb_file")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in set(word_map.keys()):
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, dims


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

''' method to save  of a dataseta and the best checkpoint. stores the following parameters:dataset_name, epoch number, number of epochs since last 
    improvement in the score, encoder model, decoder model, encoder_optimizer, decoder_optimizer,bleu4 score of this checkpoint, is this the best checkpoint '''
def save_checkpoint(data_name, epoch_number, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,bleu4, is_best):
    state = {'epoch': epoch_number,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    #creating a .tar  to store
    filename_checkpoint = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename_checkpoint)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename_checkpoint)

# class that has methods to reset and update averages.
class AverageMeter(object):
    # initialising count, sum, average to 0
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # method to dynamically update the average.
    def update(self, val):
        self.val = val
        self.sum += val 
        self.count += 1
        self.avg = self.sum / self.count

    # method to reset all parameters
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    

#update the learning rate for an optimizer for corresponding adjust factor which should be in (0,1)
def update_learning_rate(optimizer, adjust_factor):
    
    assert(0<=adjust_factor<=1),"adjust factor out of (0,1)"
    for i in optimizer.param_groups:
        i['lr'] = i['lr'] * adjust_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

#  calculate top-k accuracy given estimated labels and true labels 
def accuracy(est, true, k):
    batch_size = true.size(0)
    _, index = est.topk(k, 1, True, True)
    correct = index.eq(true.view(-1, 1).expand_as(index))
    correct_total = correct.view(-1).float().sum() 
    return correct_total.item() * (100.0 / batch_size)
