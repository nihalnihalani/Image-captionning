''';==========================================
; Title:  create_input_files_flickr30k.py
; Author: Ashkan Kazemi
; Last Updated: 31 Oct 2018
;==========================================
'''
from utilities import createInput

# Call createInput for train, val, test for given flikr30k
if __name__ == '__main__':
    createInput('flikr30k','../caption data/flikr30k.json','/media/ssd/caption data/',3,5,'/media/ssd/caption data/',50)
