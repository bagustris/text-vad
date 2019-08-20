#!/usr/bin/env python3

"""
Author: Doris Zhou
Modified by B.T. Atmaja (btatmaja@gmail.com)
Date: September 29, 2017
Performs sentiment analysis on a text file using ANEW.
Parameters:
    --dir [path of directory]
        specifies directory of files to analyze
    --file [path of text file]
        specifies location of specific file to analyze
    --out [path of directory]
        specifies directory to create output files
    --mode [mode]
        takes either "median" or "mean"; determines which is used to calculate sentence sentiment values
NOTE: Input file should one utterance per line, as it is intended.
"""
# add parameter to exclude duplicates? also mean or median analysis

import csv
import sys
import os
import statistics
import time
import argparse
import numpy as np
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('../../stanford-corenlp-full-2018-02-27')

import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()
stops = set(stopwords.words("english"))
#anew = "../lib/vad-nrc.csv"
anew = "../lib/EnglishShortened.csv"
avg_V = 5.06    # average V from ANEW dict
avg_A = 4.21
avg_D = 5.18

# performs sentiment analysis on inputFile using the ANEW database, outputting results to a new CSV file in outputDir
def analyzefile(input_file, output_dir, mode):
    """
    Performs sentiment analysis on the text file given as input using the ANEW database.
    Outputs results to a new CSV file in output_dir.
    :param input_file: path of .txt file to analyze
    :param output_dir: path of directory to create new output file
    :param mode: determines how sentiment values for a sentence are computed (median or mean)
    :return:
    """
    output_file = os.path.join(output_dir, os.path.basename(input_file).rstrip('.txt') + ".csv") #"OutputAnewSentiment_" + 
    # make buffer for list of utterance
    utterances = []
    # read file into string
    with open(input_file, 'r') as myfile:
        #for line in myfile.readlines():
            #utterance = tokenize.word_tokenize(line)
            #utterances = np.append(utterances, utterance)
            #utterances.append(utterance)
        
        # writing file
        i = 1 # to store sentence/line index
        # check each word in sentence/line for sentiment and write to output_file
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['Sentence ID', 'Sentence', 'Valence', 'Arousal', 'Dominance', 'Sentiment Label',
                          'Average VAD', '# Words Found', 'Found Words', 'All Words']
            writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=fieldnames)
            writer.writeheader()

            # analyze each sentence/line for sentiment
            for line in myfile.readlines():
                s = tokenize.word_tokenize(line.lower())
                #print("S" + str(i) +": " + s)
                all_words = []
                found_words = []
                total_words = 0
                v_list = []  # holds valence scores
                a_list = []  # holds arousal scores
                d_list = []  # holds dominance scores

                # search for each valid word's sentiment in ANEW
                words = nltk.pos_tag(s)
                for index, p in enumerate(words):
                    # don't process stops or words w/ punctuation
                    w = p[0]
                    pos = p[1]
                    if w in stops or not w.isalpha():
                        continue

                    # check for negation in 3 words before current word
                    j = index-1
                    neg = False
                    while j >= 0 and j >= index-3:
                        if words[j][0] == 'not' or words[j][0] == 'no' or words[j][0] == 'n\'t':
                            neg = True
                            break
                        j -= 1
    
                    # lemmatize word based on pos
                    if pos[0] == 'N' or pos[0] == 'V':
                        lemma = lmtzr.lemmatize(w, pos=pos[0].lower())
                    else:
                        lemma = w
    
                    all_words.append(lemma)
    
                    # search for lemmatized word in ANEW
                    with open(anew) as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            if row['Word'].casefold() == lemma.casefold():
                                if neg:
                                    found_words.append("neg-"+lemma)
                                else:
                                    found_words.append(lemma)
                                v = float(row['valence'])
                                a = float(row['arousal'])
                                d = float(row['dominance'])

                                if neg:
                                    # reverse polarity for this word
                                    v = 5 - (v - 5)
                                    a = 5 - (a - 5)
                                    d = 5 - (d - 5)

                                v_list.append(v)
                                a_list.append(a)
                                d_list.append(d)

                if len(found_words) == 0:  # no words found in ANEW for this sentence
                    writer.writerow({'Sentence ID': i,
                                     'Sentence': s,
                                     'Valence': np.nan,
                                     'Sentiment Label': np.nan,
                                     'Arousal': np.nan,
                                     'Dominance': np.nan,
                                     'Average VAD': np.nan,
                                     '# Words Found': 0,
                                     'Found Words': np.nan,
                                     'All Words': all_words
                                     })
                    i += 1
                else:  # output sentiment info for this sentence
                    # get values
                    if mode == 'median':
                        sentiment = statistics.median(v_list)
                        arousal = statistics.median(a_list)
                        dominance = statistics.median(d_list)
                    elif mode == 'mean':
                        sentiment = statistics.mean(v_list)
                        arousal = statistics.mean(a_list)
                        dominance = statistics.mean(d_list)
                    elif mode == 'mika':
                        # calculate valence
                        if statistics.mean(v_list) < avg_V:
                            sentiment = max(v_list) - avg_V
                        elif max(v_list) < avg_V:
                            sentiment = avg_V - min(v_list)
                        else:
                            sentiment = max(v_list) - min(v_list)
                        # calculate arousal
                        if statistics.mean(a_list) < avg_A:
                            arousal = max(a_list) - avg_A
                        elif max(a_list) < avg_A:
                            arousal = avg_A - min(a_list)
                        else:
                            arousal = max(a_list) - min(a_list)
                        # calculate dominance
                        if statistics.mean(d_list) < avg_D:
                            dominance = max(d_list) - avg_D
                        elif max(d_list) < avg_D:
                            dominance = avg_D - min(a_list)
                        else:
                            dominance = max(d_list) - min(d_list)
                    else:
                        raise Exception('Unknown mode')
                        
                     # set sentiment label
                    label = 'neutral'
                    if sentiment > 6:
                        label = 'positive'
                    elif sentiment < 4:
                        label = 'negative'

                    writer.writerow({'Sentence ID': i,
                                     'Sentence': s,
                                     'Valence': sentiment,
                                     'Arousal': arousal,
                                     'Dominance': dominance,
                                     'Average VAD': np.mean([sentiment, arousal, dominance]),
                                     'Sentiment Label': label,
                                     '# Words Found': ("%d out of %d" % (len(found_words), len(all_words))),
                                     'Found Words': found_words,
                                     'All Words': all_words
                                     })
                    i += 1


def main(input_file, input_dir, output_dir, mode):
    """
    Runs analyzefile on the appropriate files, provided that the input paths are valid.
    :param input_file:
    :param input_dir:
    :param output_dir:
    :param mode:
    :return:
    """

    if len(output_dir) < 0 or not os.path.exists(output_dir):  # empty output
        print('No output directory specified, or path does not exist')
        sys.exit(0)
    elif len(input_file) == 0 and len(input_dir)  == 0:  # empty input
        print('No input specified. Please give either a single file or a directory of files to analyze.')
        sys.exit(1)
    elif len(input_file) > 0:  # handle single file
        if os.path.exists(input_file):
            analyzefile(input_file, output_dir, mode)
        else:
            print('Input file "' + input_file + '" is invalid.')
            sys.exit(0)
    elif len(input_dir) > 0:  # handle directory
        if os.path.isdir(input_dir):
            directory = os.fsencode(input_dir)
            for file in os.listdir(directory):
                filename = os.path.join(input_dir, os.fsdecode(file))
                if filename.endswith(".txt"):
                    start_time = time.time()
                    print("Starting sentiment analysis of " + filename + "...")
                    analyzefile(filename, output_dir, mode)
                    print("Finished analyzing " + filename + " in " + str((time.time() - start_time)) + " seconds")
        else:
            print('Input directory "' + input_dir + '" is invalid.')
            sys.exit(0)


if __name__ == '__main__':
    # get arguments below:
    #input_file = '../data/iemocap_text_10036.txt'
    input_file = '../data/emobank_text.txt'
    #input_file = './input.txt'
    input_dir = ''#only for input directory
    mode = 'mika'
    output_dir = '../out/anew_' + mode
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    # run main with arguments above
    sys.exit(main(input_file, input_dir, output_dir, mode))
