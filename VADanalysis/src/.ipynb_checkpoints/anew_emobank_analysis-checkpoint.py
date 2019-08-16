# emobank analysis from textfile
import numpy as np 
import pandas as pd
from nltk import tokenize

data_path = '/media/bagus/data01/github/text-vad/VADanalysis/data/'
counter = 0 
utterances = []
with open(data_path + 'emobank_text.txt', 'r') as myfile: 
    for line in myfile.readlines(): 
        utterance = tokenize.word_tokenize(line)
        #utterances.append(utterance)
    
    #for s in utterances[:5]:
        print(utterance.lower())
        #words = nlp.pos_tag(s.lower())
        #print(s.lower)