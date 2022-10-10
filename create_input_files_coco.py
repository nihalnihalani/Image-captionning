''';==========================================
; Title:  create_input_files_coco.py
; Author: Ashkan Kazemi
; Last Updated: 31 Oct 2018
;==========================================
'''
from utilities import createInput

# Call createInput for train, val, test for given coco
if __name__ == '__main__':
    createInput('coco','../caption data/coco.json','/media/ssd/caption data/',5,5,'/media/ssd/caption data/',50)
