''';==========================================
; Title:  datasets.py
; Author: Lakshmi Venkatesh Kakumani
; Last Updated: 6 Nov 2018
;==========================================
'''
import torch
from torch.utils.data import Dataset
import h5py
import json
import os

# Create batches in pyTorch DataLoader
class CreateCaptionDataset(Dataset):
   
    def __init__(self, dataset_folder, dataset_name, split, img_transform=None):
        
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file 
        self.h = h5py.File(os.path.join(dataset_folder, self.split + '_IMAGES_' + dataset_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load caption lengths
        with open(os.path.join(dataset_folder, self.split + '_CAPLENS_' + dataset_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
		
		# Load captions 
        with open(os.path.join(dataset_folder, self.split + '_CAPTIONS_' + dataset_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        self.img_transform = img_transform
        self.dataset_size = len(self.captions)
	
	# retrieve nth caption for the corresponding nth image
    def __getCaptionItem__(self, i):
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.img_transform is not None:
            img = self.img_transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
