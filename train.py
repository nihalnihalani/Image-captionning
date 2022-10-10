''';==========================================
; Title:  train.py
; Author: Lakshmi Venkatesh Kakumani 
; Last Updated: 07 Dec 2018
;==========================================
'''

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utilities import *
from nltk.translate.bleu_score import corpus_bleu

data_folder = './Images/'  # folder with data files saved by create_input_files.py
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'  # used to train flickr8k - change it to flickr30k or coco for these datasets

embedded_dim = 512  # embedded dimension
attention_dim = 512  # attention network hidden neurons
decoder_dim = 512  # decoder dimension
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # converts tensors to CUDA variables if gpu is available  

start_epoch = 0
epochs = 120  # Total number of epochs
epochs_since_improvement = 0  # keep track of epochs since the last best score...needed for early stopping
batch_size = 32
workers = 1  # for data-loading
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients to avoid gradients vanishing
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention' as in the paper
best_bleu4 = 0.  # best BLEU-4 score is stored here
print_freq = 100  
fine_tune_encoder = False  
checkpoint = None #path to checkpoint


def main():
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json') # Loading the wordmap file using dataname(flickr8k)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

     if checkpoint is None: # if there is no checkpoint
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=embedded_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout) # using the archi from models file
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr) # Adam optimizer
        encoder = Encoder() # using the archi from models file
        encoder.fine_tune(fine_tune_encoder) # finetune the encoder
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None # Adam optimizer

    else: # load the checkpoint file to continue training 
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)
    decoder = decoder.to(device) # converts tensors to CUDA variables if gpu is available  
    encoder = encoder.to(device) # converts tensors to CUDA variables if gpu is available  
    criterion = nn.CrossEntropyLoss().to(device) # Loss function
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # normalizing the data
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True) # Data loader for train set
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True) # Data loader for val set

    # Training Starts !!!!!!!
    for epoch in range(start_epoch, epochs):

        if epochs_since_improvement == 20: # Early stopping if the BLEU scores degrade for a long time
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8) # learning rate decay to help the training process
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8) # learning rate decay to help the training process
	train(train_loader=train_loader, # Training using the encoder, decoder archi, input images, loss function and optimizers
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)
        recent_bleu4 = validate(val_loader=val_loader, # Validation after every epoch
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1 # If no improvement
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, # Save checkpoint after every epoch
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
   
    decoder.train()  
    encoder.train()

    losses = AverageMeter()  #Loss
    top5accs = AverageMeter() # Top-5 accuracy 

    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader): # sending batches for training( mini-batch gradient descent)

        imgs = imgs.to(device) # converts tensors to CUDA variables if gpu is available  
        caps = caps.to(device) # converts tensors to CUDA variables if gpu is available  
        caplens = caplens.to(device) # converts tensors to CUDA variables if gpu is available  

        imgs = encoder(imgs) # Encoder output for input image (ResNet-101 architecture)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens) # Decoder output

        targets = caps_sorted[:, 1:] # Removing the <start> tag

        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = criterion(scores, targets) # Loss calculated
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean() # Regularization as stated in the paper

        decoder_optimizer.zero_grad() 
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward() # Backward propagation

        if grad_clip is not None: # Gradient clipping
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step() # Update the weights of the decoder
        if encoder_optimizer is not None:
            encoder_optimizer.step() # Update the weights of the encoder

        top5 = accuracy(scores, targets, 5) # Metrics
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader), loss=losses,top5=top5accs))
                                                                                                                                                   
                                                                          

def validate(val_loader, encoder, decoder, criterion):
    decoder.eval()  # eval mode 
    if encoder is not None:
        encoder.eval()

    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Same code as training but no back propagation
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        if encoder is not None:
            imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        targets = caps_sorted[:, 1:]

        scores_copy = scores.clone()
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = criterion(scores, targets)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        losses.update(loss.item(), sum(decode_lengths))
        top5 = accuracy(scores, targets, 5)
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),loss=losses, top5=top5accs))

	# Storing the references for the image
        allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and <pad>
            references.append(img_captions)

        _, preds = torch.max(scores_copy, dim=2) # Getting the predictions
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  
        preds = temp_preds
        hypotheses.extend(preds)

        assert len(references) == len(hypotheses)

    bleu4 = corpus_bleu(references, hypotheses) # Calculate BLEU-4 scores

    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
