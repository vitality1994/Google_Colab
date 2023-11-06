# import necessary packages
import sys
import argparse

import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends, flatten
from nltk.lm import MLE
from nltk.util import everygrams
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.lm import MLE, Laplace
from nltk.lm.models import StupidBackoff, WittenBellInterpolated

nltk.download('punkt')

import re
import string


# arguments setting -----------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-test', required=False, default='no_test')
args, unkown = parser.parse_known_args()

# to read authorlist.txt file contain list of file names to read.
try:
  directory_name = sys.argv[1]
  path = f'./{directory_name}.txt'

except:
  print('Please psss the directory_name')
  

f = open(path, 'r')
authors = f.read()

authors_list = authors.split('\n')

# -----------------------------------------------------------------------------

# necessary functions for training/testing --------------------
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    return text

def preprocess_text(author_input):

  for author in authors_list:
    author_name = author.split('_')[0]
    if author_name==author_input:
      f = open(f'./ngram_authorship_train/{author_name}.txt', 'r')
      text = f.read()

  sents = sent_tokenize(text, language='english')

  clean_sents = []
  for sent in sents:
    clean_sents.append(clean_text(sent))

  clean_words = []
  for sent in clean_sents:
    clean_words.append(word_tokenize(sent))

  return clean_words

def split_train_test(clean_words):
  train_size = len(clean_words) * 0.9
  train_sentences = clean_words[:int(train_size)]
  test_sentences = clean_words[int(train_size):]
  return train_sentences, test_sentences

def train_lm(train_words, n, method):
  train, vocab = padded_everygram_pipeline(n, train_words)

  if method == 'Laplace':

    lm = Laplace(n)

  elif method == 'MLE':

    lm = MLE(n)

  elif method == 'StupidBackoff':

    lm = StupidBackoff(order = n)

  elif method == 'WittenBellInterpolated':

    lm = WittenBellInterpolated(order = n)

  lm.fit(train, vocab)

  return lm

def test_lm(lm, test_words, n):
  test_paddedline  = list(everygrams(list(pad_both_ends(test_words, n=n)), max_len=n))
  return lm.perplexity(test_paddedline)

def show_accuracy_lm(author, n):

  if author == 'austen':
    test_set = austen_test

  elif author == 'dickens':
    test_set = dickens_test

  elif author == 'tolstoy':
    test_set = tolstoy_test

  elif author == 'wilde':
    test_set = wilde_test


  true_or_false_list = []

  for test_sent in test_set:

    austen_perplexity = test_lm(austen_lm, test_sent, n)
    dickens_perplexcity = test_lm(dickens_lm, test_sent, n)
    tolstoy_perplexity = test_lm(tolstoy_lm, test_sent, n)
    wilde_perplexity = test_lm(wilde_lm, test_sent, n)

    compares = {'austen': austen_perplexity,
                'dickens': dickens_perplexcity,
                'tolstoy': tolstoy_perplexity,
                'wilde': wilde_perplexity}

    min_perplexity = min(compares, key=compares.get)


    if author == min_perplexity:

      true_or_false_list.append(1)

    else:
      true_or_false_list.append(0)


  return (sum(true_or_false_list)/len(true_or_false_list))*100
# --------------------------------------------------------------


# Preprocessing text data
austen_clean_words = preprocess_text('austen')
dickens_clean_words = preprocess_text('dickens')
tolstoy_clean_words = preprocess_text('tolstoy')
wilde_clean_words = preprocess_text('wilde')



# start experiments and predictions --------------------------------------------

# experiments using training / development datasets

# lists for experiments with different type of models...
# n_list = [2, 3]
# prob_method_list = ['Laplace', 'MLE', 'StupidBackoff', 'WittenBellInterpolated']

# for the best model
n_list = [2]
prob_method_list = ['Laplace']

if args.test == 'no_test':
  # data preparation
  print('splitting into training and development...')
  austen_train, austen_test = split_train_test(austen_clean_words)
  dickens_train, dickens_test = split_train_test(dickens_clean_words)
  tolstoy_train, tolstoy_test = split_train_test(tolstoy_clean_words)
  wilde_train, wilde_test = split_train_test(wilde_clean_words)

  for n in n_list:

    if len(n_list)!=1:
      print(f'Experiments with {n}-gram LM')

    for prob_method in prob_method_list:
      
      if len(n_list)==1:
        print(f'training LMs... (this may take a while)')
      else:
        print(f'training LMs... (this may take a while) / Using {prob_method}')

      austen_lm = train_lm(austen_train, n, method = prob_method)
      dickens_lm = train_lm(dickens_train, n, method = prob_method)
      tolstoy_lm = train_lm(tolstoy_train, n, method = prob_method)
      wilde_lm = train_lm(wilde_train, n, method = prob_method)

      austen_rate = show_accuracy_lm('austen', n)
      dickens_rate = show_accuracy_lm('dickens', n)
      tolstoy_rate = show_accuracy_lm('tolstoy', n)
      wilde_rate = show_accuracy_lm('wilde', n)

      print('Results on dev set:')
      print(f'austen: {round(austen_rate,1)}% correct')
      print(f'dickens: {round(dickens_rate, 1)}% correct')
      print(f'tolstoy: {round(tolstoy_rate, 1)}% correct')
      print(f'wilde: {round(wilde_rate, 1)}% correct')

  # # for text generation
  # prompt = [['you', 'are'], ['I', 'love'], ['The', 'weather'], ['Beautiful', 'world'], ['kindness', 'of']]

  # for i in prompt:

  #   austen_generated = austen_lm.generate(6, text_seed=i)
  #   dickens_generated = dickens_lm.generate(6, text_seed=i)
  #   tolstoy_generated = tolstoy_lm.generate(6, text_seed=i)
  #   wilde_generated = wilde_lm.generate(6, text_seed=i)

  #   print('austen', i, austen_generated, round(test_lm(austen_lm, i+austen_generated, n), 2))
  #   print('dickens', i, dickens_generated, round(test_lm(dickens_lm, i+dickens_generated, n), 2))
  #   print( 'tolstoy', i, tolstoy_generated, round(test_lm(tolstoy_lm, i+tolstoy_generated, n), 2))
  #   print('wilde', i, wilde_generated, round(test_lm(wilde_lm, i+wilde_generated, n), 2))

  #   print()




# make predictions with testfile
elif args.test == 'testfile':


# Preprocessing text data

  n=2

  # train with Laplace and test
  print('training LMs... (this may take a while)')
  austen_lm = train_lm(austen_clean_words, n, method='Laplace')
  dickens_lm = train_lm(dickens_clean_words, n, method='Laplace')
  tolstoy_lm = train_lm(tolstoy_clean_words, n, method='Laplace')
  wilde_lm = train_lm(wilde_clean_words, n, method='Laplace')

  # testfile pre-processing
  f = open(f'./testfile.txt')
  text = f.read()
  
  sents = sent_tokenize(text, language='english')

  clean_sents = []
  for sent in sents:
    clean_sents.append(clean_text(sent))

  clean_words = []
  for sent in clean_sents:
    clean_words.append(word_tokenize(sent))


  # classifications

  for test_sent in clean_words:

    austen_perplexity = test_lm(austen_lm, test_sent, n)
    dickens_perplexcity = test_lm(dickens_lm, test_sent, n)
    tolstoy_perplexity = test_lm(tolstoy_lm, test_sent, n)
    wilde_perplexity = test_lm(wilde_lm, test_sent, n)

    compares = {'austen': austen_perplexity,
                'dickens': dickens_perplexcity,
                'tolstoy': tolstoy_perplexity,
                'wilde': wilde_perplexity}

    min_perplexity = min(compares, key=compares.get)

    print(min_perplexity)





