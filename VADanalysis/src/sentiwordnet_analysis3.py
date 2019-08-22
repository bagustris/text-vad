# sentiment analysis with sentiwordnet
# NOTE: to be run using bash shell
# usage: python3.6 sentiwordnet_analysis.py > output.txt

import csv
import sys
import os
import time
import argparse
import statistics
import nltk

import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag 

lemmatizer = WordNetLemmatizer()

avg_V = 0.01

def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def clean_text(text):
    text = text.replace("<br />", " ")
    #text = text.decode("utf-8")
    return text
 
def swn_polarity(text):
    """
    Return a sentiment polarity/valence, range=(-1, 1)
    """
    
    tokens_count = 0
    sentiment = 0.0
    sentiment_list = []
    tagged_sentence = pos_tag(word_tokenize(text))
    
    for word, tag in tagged_sentence:
        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue
        lemma = lemmatizer.lemmatize(word, pos=wn_tag) #else if wn_tag not in 
        if not lemma:
            continue
        synsets = wn.synsets(lemma, pos=wn_tag)  #else if not lemma
        if not synsets:
            continue
        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        # substract pos by neg. In SWN, total pos+neg+obj = 1
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        sentiment_list.append(sentiment)
        tokens_count += 1

    #print(sentiment/tokens_count)
    if mode == 'mean':
        # if no tokens found should be neutral
        if not tokens_count:
        #print('0.0')
            return 0
        # return mean
        return sentiment/tokens_count
    elif mode == 'median':
        if not tokens_count:
        #print('0.0')
            return 0
        return statistics.median(sentiment_list)
    elif mode == 'mika':
        if not tokens_count:
            return 0
        else:
            if statistics.mean(sentiment_list) < avg_V:
                sentiment = max(sentiment_list) - avg_V
            elif max(sentiment_list) < avg_V:
                sentiment = avg_V - min(sentiment_list)
            else:
                sentiment = max(sentiment_list) - min(sentiment_list)
            return sentiment
    else:
        raise Exception('Unknown mode')


def main(input_file, output_dir, mode):   
    output_file = os.path.join(output_dir, os.path.basename(input_file).rstrip('.txt') + ".csv") 
     
    # read file into string
    with open(input_file, 'r') as myfile:
        fulltext = myfile.readlines()
        
    with open(output_file, 'w', newline='') as f:
        for text in fulltext:   
            score = swn_polarity(text)
            f.write(str(score)+'\n')

if __name__ == '__main__':
    input_file = '../data/iemocap_text_10036.txt'
    #input_file = '../data/emobank_text.txt'
    #input_file = './input.txt'
    input_dir = ''#only for input directory
    mode = 'mean'
    output_dir = '../out/senti_' + mode
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    # run main with arguments above
    sys.exit(main(input_file, output_dir, mode))
