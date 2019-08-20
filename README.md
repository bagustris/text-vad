# Text Emotion Recognition and Sentiment Analysis based on Affective Dictionary/Lexicon

## Technical details
### Requirement
This script required the following packages:
- nltk
- stanfordcorelp

You also need to download nltk stopword, punkt and wordnet as follow

    >>> import nltk
    >>> nltk.dowload('stopwords', 'punkt', 'wordnet')

### Example Usage
    $ python3.6 anew_vad_analysis.py
    $ python3.6 sentiwordnet_analysis3.py
    $ python3.6 sentiwordnet_analysis3
    
Specify input text file (input also can be directory of text file), mode and output directory on that python files. Available mode are: 'mean', 'median', and 'mika' (Mika et al., 2016).

Working directory is `./src`, input file is targetted in `data`, and output file will be located in `out` directory. Example of file `input.txt` is included in `src` directory.

### Directory structure
```
.
├── stanford-corenlp-full-XXXX-XX-XX (downloaded separately)
└── VADanalysis
    ├── data
    ├── lib
    ├── out
    └── src
```
### Database and VAD scale
 - IEMOCAP and Emobank database uses 1-5 scale (negative to positive) for each valence, arousal and doiminance.
 - ANEW use 1-9 scale (positive to negative) for each valence, arousal and dominance score.
 - Sentiwordnet use (-1, 1) valence/sentiment scale.
 - VADER uses (-1, 1) valence/sentiment scale (compound).

## Theoritical details
Sentiment analysis is an automated task to automatically evaluate the overall sentiment evoked by a text – positive or negative. The value determining this sentiment is called valence.

There are many existing tools and models for sentiment analysis, and we have implemented and summarized some, listed below. We only focusing on valence, arousal, dominance (vad) score determination rather than categorical model as it is strongly closed to human behaviour.

### ANEW

ANEW, short for Affective Norms for English Words, is a database of 1,034 English words that have been manually rated by many human volunteers on three affective measures: pleasure (valence), arousal (excitement), and dominance (level of control), as elicited by a particular word (Bradley & Lang 1999). In 2013, Warriner et al. expanded the database to nearly 14,000 English lemmas, and also split data by gender, age, and educational differences in raters (Warriner et al. 2013). In both databases, affective ratings are on a scale from 1 to 9, where 1 is the least pleasurable/exciting/controlling, and 9 is the most.

In our implementation, we used Warriner et al.’s expanded database, extracting the average valence, pleasant, and arousal for each word from the larger database, for word-by-word sentiment analysis.

We wrote a Python script to perform sentiment analysis with the resulting data. Given a body of text in .txt format, we first tokenized the text into sentences using the NLTK’s sentence tokenizer, and then tokenized each sentence into individual words with the NLTK’s word tokenizer, stripping out all stop words found in the NLTK’s English stop word database. For each non-stop word in each sentence, we searched for the word in the database and stored its individual valence, arousal, and dominance values.

As ANEW uses ratings from a scale of 1 (most negative) to 9 (most positive), valence values of 5 are considered neutral; values less than 5 are considered positive, and values greater than 5 are considered negative. In accordance with Hutto & Gilbert (2014)’s method for accounting for negative values, if a word in the three words prior to the word indicated negation – “not” or “no” – we reversed the polarity of that word. We did this by computing (5 – (valence – 5)) as the new valence value.

After finding sentiment ratings for each non-stop word in each sentence, we found overall sentiment ratings for the sentence by either taking the median or the mean of the sentiment ratings for each word in that sentence, according to the method selected by the user. For each sentence, we labeled the sentence’s valence as negative if less than 5, neutral if equal to 5, and positive if greater than 5.
Weaknesses of this approach: As this is a word-for-word approach to analyzing the sentiment of an entire sentence, the results are limited by the number of words available in ANEW.

### SENTIWORDNET
SENTIWORDNET (Baccianella et al., 2009) is a lexical resource explicitly devised for supporting sentiment classification and opinion mining applications. The version 3.0 is an improvement of previous version 1.0. SENTIWORDNET consist of three score: positive, negative and objective neutral, while the dictionary it self only lists positive and negative score. The objective score then is calculated as, 

**Obj = 1 - (Pos + Neg)**

In this research, valence score for each word is subtraction of positive (Pos) by negative (Neg) score,

**V = Pos - Neg**

Valence score for each utterance then is calculated using either mean, median or Mika method. As it used WordNet as dictionary, a sets of synonyms is used to search the similar word meaning. I used degree of synset "0" to take the most common of a word given its part-of-speech (POS) in WordNet. Sentence with no word found in SENTIWORDNET is categorized as neutral.

### NLTK VADER
NLTK, an abbreviation for the Natural Language Toolkit, is a robust library of Python functions for various natural language processing tasks (Bird 2006). Among its many functions is an implementation of the VADER (Valence Aware Dictionary for sEntiment Reasoning) sentiment analysis tools.

VADER is a simple rule-based model for sentiment analysis for general sentiment analysis. It is most accurate for social media data, but is generalizable to other domains as well. To create the model, Hutto & Gilbert first constructed a gold-standard list of features using features from several widely used sentiments lexicons and some of their own (such as emoticons), using a wisdom-of-the-crowd approach to acquire a valid point estimate for the valence of each feature, as well as intensity ratings from Amazon Mechanical Turk workers. They also created five general heuristics for sentiment analysis of texts: punctuation, capitalization, intensifiers, the use of the contrastive conjunction but, and negation.

We implemented VADER in Python using NLTK’s VADER library. Identically to our implementation of ANEW, we tokenized texts into sentences; for each sentence, we used NLTK’s SentimentIntensityAnalyzer to obtain polarity scores for that sentence. Scores are normalized on a scale from 1 to -1, where positive values have a positive valence, 0 is neutral, and negative values have a negative valence. Sentence without word found in VADER library is also categorized as neutral.

### References
Bird, S. (2006, July). NLTK: the natural language toolkit. In Proceedings of the COLING/ACL
on Interactive presentation sessions (pp. 69-72). Association for Computational
Linguistics.

Bradley, M. M., & Lang, P. J. (1999). Affective norms for English words (ANEW): Instruction
manual and affective ratings (pp. 1-45). Technical report C-1, the center for research in
psychophysiology, University of Florida.

Hutto, C. J., & Gilbert, E. (2014, May). Vader: A parsimonious rule-based model for sentiment
analysis of social media text. In Eighth international AAAI conference on weblogs and
social media.

Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and
dominance for 13,915 English lemmas. Behavior research methods, 45(4), 1191-1207.

Mäntylä, M., Adams, B., Destefanis, G., Graziotin, D., & Ortu, M. (2016). Mining Valence, Arousal, and Dominance - Possibilities for Detecting Burnout and Productivity? https://doi.org/10.1145/2901739.2901752

https://github.com/dwzhou/SentimentAnalysis

### Contributor
@bagustris

#### Please cite this work if you use the code inside this repository (an accompany paper may added later).
