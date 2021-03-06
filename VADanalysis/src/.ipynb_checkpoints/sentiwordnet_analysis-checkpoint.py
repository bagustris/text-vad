# sentiment analysis with sentiwordnet
# usage: python3.6 sentiwordnet_analysis.py > output.txt
# this version is obsolete, please use sentiwordnet_analysis2.

import csv
import sys
import os
import time
import argparse

import nltk

import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag 

lemmatizer = WordNetLemmatizer()

# change input_file here
#input_file = '../data/iemocap_text_10036.txt'
input_file = 'input.txt'

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
    #text = clean_text(text)
 
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        #print(raw_sentence)
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))

        for word, tag in tagged_sentence:
            
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
 
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
 
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
 
            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
 
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            
            tokens_count += 1
            
    
    if not tokens_count:
        return 0
    
    #print(sentiment/len(tokens_count))
    return sentiment
    
def main(input_file):
    """
    Performs sentiment analysis on the text file given as input using the sentiwordnet database.
    Outputs results to a new CSV file in output_dir.
    :param input_file: path of .txt file to analyze
    :param output_dir: path of directory to create new output file
    :return:
    """
    

    # read file into string
    with open(input_file, 'r') as myfile:
        fulltext = myfile.readlines()
        
    score = [swn_polarity(text) for text in fulltext]

if __name__ == '__main__':
    main(input_file)
