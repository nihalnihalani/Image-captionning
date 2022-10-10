''';==========================================
; Title:  train.py
; Author: Chockalingam Ravi Sundaram 
; Last Updated: 10 Dec 2018
;==========================================
'''
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utilities import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

# Set the Directory variables for data, checkpoint and word map
data_folder = '/media/ssd/caption data'  # folder with data files saved by create_input_files.py
checkpoint = '../BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'  # Trained model
word_map = '/media/ssd/caption data/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json' # Vocabulary of words needed to generate captions for the given dataset (Replace flickr8k with flickr30k or coco to use wordmaps of other datasets)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets the tensors to CUDA variables if gpu is available

#Loading the trained model for evaluation and beam search
print('Loading checkpoint!!!')
model = torch.load(checkpoint)
decoder = model['decoder']
decoder = decoder.to(device) # Converts the decoder model to CUDA variables if gpu is available
decoder.eval() # No back-prop used as the decoder is only used for evaluation purposes
encoder = model['encoder']
encoder = encoder.to(device) # Converts the encoder model to CUDA variables if gpu is available
encoder.eval() # No back-prop used as the encoder is only used for evaluation purposes

# Load word map
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map) # Total number of words in the word map

# Data Loading and Pre-processing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])# Normalization transform for input images

loader = torch.utils.data.DataLoader(CaptionDataset(data_folder, 'flickr8k', 'VAL', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True) # Data Loader

# Evaluation code starts !!!
print('Evaluation begins')
ref = list()
hyp = list()
beam_size = 5

for i,(image, captions, caption_lens, all_captions) in enumerate(loader):
	image = image.to(device)  # Move to GPU if available
	k = beam_size

	# Encoder output
	encoder_out = encoder(image)
	encoder_img_size = encoder_out.size(1)
	encoder_dim = encoder_out.size(3)
	
	encoder_out = encoder_out.view(1, -1, encoder_dim)  # Flatten encoding output as 1 x num_of_pixels x enc_dim
	num_pixels = encoder_out.size(1)
	
	encoder_out = encoder_out.expand(k, num_pixels, encoder_dim) # Since the beam_size=k, we will treat the problem as having a batch size of k
	k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device) # Tenssor to store the top 'k' previous words at each step in the decoder (Now they are initialized with <start>)
	seqs = k_prev_words

	top_k_scores = torch.zeros(k, 1).to(device) # tensor to store the scores of the sequences (Now they are initialized with zeros)
	complete_seqs = list()
	complete_seqs_scores = list()

        # Decoding code starts
	step = 1
	h, c = decoder.init_hidden_state(encoder_out)
	smth_wrong=False

	while True:
		embeddings = decoder.embedding(k_prev_words).squeeze(1)  # Embedding layer output

        	att_weighted_enc, _ = decoder.attention(encoder_out, h)  # Attention network output to get the weighed encoding

            	gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar in LSTM cell
            	att_weighted_enc = gate * att_weighted_enc

            	h, c = decoder.decode_step(torch.cat([embeddings, att_weighted_enc], dim=1), (h, c)) # LSTM cell output with Attention network input concatenated with the Embedding layer output 

            	scores = decoder.fc(h)  
            	scores = F.log_softmax(scores, dim=1) # Score = FC Layer + Softmax 
            	scores = top_k_scores.expand_as(scores) + scores 

		if step == 1: # Only for the first step, all k points have the same score
                	top_k_scores, top_k_words = scores[0].topk(k, 0, True, True) 
            	else:
                	top_k_scores, top_k_inds = scores.view(-1).topk(k, 0, True, True) # Finding the top k scores and their respective indices
		prev_word_inds = top_k_inds / vocab_size  
		next_word_inds = top_k_inds % vocab_size # Getting the index of the next word in the Word map(or vocab)

		seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1) # Adding new words to the sequences 
		incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']] # List of sequences which are incomplete (i.e did not reach <end>)
		complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds)) 

		if len(complete_inds) > 0:
			complete_seqs.extend(seqs[complete_inds].tolist())
			complete_seqs_scores.extend(top_k_scores[complete_inds])
		k = k-len(complete_inds)  # reduce beam length accordingly
		
		if k == 0: # Stop when you are done generating sequences
			break
		# Proceed with incomplete sequences
		seqs = seqs[incomplete_inds]
		# Getting the previous states of the LSTM networks at the incomplete indices
		h = h[prev_word_inds[incomplete_inds]] 
		c = c[prev_word_inds[incomplete_inds]]
		encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
		top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
		k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

		if step > 50: # Stop if things are going on for a long time!!!
			smth_wrong = True
			break
		step += 1
	if smth_wrong is not True:
		i = complete_seqs_scores.index(max(complete_seqs_scores)) # Choosing the sequence with the highest score
		seq = complete_seqs[i]
		sentence = [rev_word_map[seq[i]] for i in range(len(seq))] # Converting the sequence from indices to words using word map
	else:
		seq = seqs[0][:20] # Terminate the sequence 
		sentence = [rev_word_map[seq[i].item()] for i in range(len(seq))] # Convert the indices to words using the word map
		sentence = sentence + ['<end>'] # Manually add '<end>' if things have been going on for a long time

	img_captions = all_captions[0].tolist() # References (or actual true captions in the dataset)
	img_captions = list(map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                           img_caps))  # remove <start>, <end> and <pad> in the captions
	references.append(img_captions) # References without any <start>, <pad>, <end>
	
	# Hypothesis consists of seq without <start>, <pad>, <end>		
	hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
	assert len(references) == len(hypotheses) # Just to check if the number of references = number of hypothesis
		
	bleu4 = corpus_bleu(references, hypotheses) # BLEU-4 score from the ntlk library
	
print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, bleu4))

		  

	

 	
















