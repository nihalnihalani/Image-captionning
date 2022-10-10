''';==========================================
; Title:  create_input_files_flickr8k.py
; Author: Ashkan Kazemi
; Last Updated: 31 Oct 2018
;==========================================
'''
from utilities import createInput

# Call createInput for train, val, test for given flikr8k
if __name__ == '__main__':
    createInput('flikr8k','../caption data/flikr8k.json','/media/ssd/caption data/',5,5,'/media/ssd/caption data/',50)
