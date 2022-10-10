''';==========================================
; Title:  models.py
; Author: Chockalingam Ravi Sundaram 
; Last Updated: 01 Dec 2018
;==========================================
'''

import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Convert tensors to cuda variables if gpu is available


class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        modules = list(resnet.children())[:-2] # Removing the linear and pooling layers of the resnet model as we need to only extract the features
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size)) # Resize image to fixed size to allow input images of variable size

        self.fine_tune()
    # defines how the inputs run sequentially through the Encoder network
    def forward(self, images): # returns the encoder output images for a given input image ( 3 x width x height)
        out = self.resnet(images)  # Resnet output
        out = self.adaptive_pool(out)  
        out = out.permute(0, 2, 3, 1)  
        return out

    def fine_tune(self, fine_tune=True): 
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]: # if fine_tune is True, it allows the weights to be updated in the conv blocks 2-4
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # FC layer to transform encoder's output image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # FC layer to transform decoder's output 
        self.full_att = nn.Linear(attention_dim, 1)  # FC layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax layer to calculate weights for the image

    def forward(self, encoder_out, decoder_hidden): # defines how the inputs run sequentially through the Attention network
        att1 = self.encoder_att(encoder_out)  
        att2 = self.decoder_att(decoder_hidden) 
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2) 
        alpha = self.softmax(att)  
        att_weighted_enc = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) 

        return att_weighted_enc, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCells
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # FC layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # FC layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # FC layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # FC layer to find scores over vocabulary
        self.init_weights()  

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1) # Random initialization of Embedding layer weights using uniform distribution
        self.fc.bias.data.fill_(0) # Initializing the biases with zeros
        self.fc.weight.data.uniform_(-0.1, 0.1) # Random initialization of FC layer weights using uniform distribution

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings) # Load pre-trained embeddings

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters(): # Fine-tune the embedding layer weights if fine-tune is True
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out) # Initial hidden state of the LSTMCell 
        c = self.init_c(mean_encoder_out) # Initial cell state of the LSTMCell based on the encoder's output image
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths): # defines how the inputs run sequentially through the Decoder network
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # Flatten encoded image
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True) # Sort input data by decreasing lengths 
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions) # Embedding layer

        h, c = self.init_hidden_state(encoder_out) # Initialize LSTM hidden and cell states

        decode_lengths = (caption_lengths - 1).tolist() # Decode length is subtracted by one as we don't decode <end> 

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device) # first initiaization with zeros
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device) # first initiaization with zeros

        for t in range(max(decode_lengths)): # For each time-step do the following:
            batch_size_t = sum([l > t for l in decode_lengths])
            att_weighted_enc, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t]) # Get the attention network's output based on the encoded image and the hidden states of the LSTMCell
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # Gating scalar
            att_weighted_enc = gate * att_weighted_enc # Gated attention-weighted-encoded image 
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])) # Updating the states of the LSTMCell for this time-step
            preds = self.fc(self.dropout(h))  # The predictions based on the states of the LSTMCell
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha # The weights of the image needed for the next time-step

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
