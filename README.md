# Image Captioning using CNNs and LSTMs with Visual Attention
Image Captioning is a much more involved task than image recognition or classification, because of the additional challenge of recognizing the interdependence between objects/concepts in an image to creating a sentence description. In this project, we implement Image Captioning using CNN and LSTMs with Visual Attention. Our model consists of three sub-models: encoder- to extract the most important features in an image, attention- to determine the parts of the image which will be useful to generate the next word in the sequence given the previous word, and decoder- which generates a word for each time step given the output of the encoder and attention network. We have implemented our project on MS COCO '14, Flickr30k, and Flickr8k datasets. The code is written in Python using the PyTorch framework.

Some of the results of our model are as follows:

![Hi](/writeup/Figure_11_dark.png)
![Hi](/writeup/Figure_1_indicates_proximity.png)
![Hi](/writeup/Figure_2_action(swinging).png)
![Hi](/writeup/Figure_4.png)
![Hi](/writeup/Figure_5.png)
![Hi](/writeup/Figure_5_number(couple).png)
![Hi](/writeup/Figure_6.png)
![Hi](/writeup/Figure_10.png)
![Hi](/writeup/Figure_7.png)
![Hi](/writeup/Figure_8.png)
![Hi](/writeup/Figure_9.png)
![Hi](/writeup/Figure_1.png)
![Hi](/writeup/Figure_2.png)
![Hi](/writeup/Figure_3.png)
