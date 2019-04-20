import cv2
import mxnet.gluon.data as data
import json
import gluonnlp
from collections import Counter
import mxnet as mx
import numpy as np
import re 

MAX_WORDS = 100
VOCAB_SIZE = 2000

def read_json(path='/home/jiaming/Downloads/dataset/msr-vtt/test_videodatainfo.json'):
    with open(path) as file:
        attribute = json.load(file)
    
    print(attribute.keys())
    print(attribute['info'].keys())
    #print(attribute['videos'][0])
    #print(len(attribute['sentences']))
    #print(attribute['info'].values())
    #print(attribute['videos'][0])
    print(attribute['sentences'][0])

    test_dict = dict()
    for item in attribute['sentences']:
        temp = item['caption'].split()
        if item['video_id'] not in test_dict:
            sentence = []
            #temp = item['caption'].split()
            sentence.append(temp)
            test_dict[item['video_id']] = sentence
        else:
            test_dict[item['video_id']].append(temp)
    
    for key in test_dict:
        max_len = np.amax([len(item) for item in test_dict[key]])
        print(len(test_dict[key]),max_len)
    # vocab = set()
    # caption_length = dict()
    # for item in attribute['sentences']:
    #     caption = item['caption']#.split()
    #     caption = caption.split()
    #     print(caption)

    #     if len(caption) in caption_length:
    #         caption_length[len(caption)] += 1
    #     else:
    #         caption_length[len(caption)] = 1

    #     for word in caption:
    #         vocab.add(word)

    #for key in caption_length:
    #    print(key,caption_length[key])

    #return vocab

def loadGloveModel(gloveFile='/home/jiaming/Downloads/dataset/glove.6B/glove.6B.50d.txt'):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def loadGloveModel_2(gloveFile='/home/jiaming/Downloads/dataset/glove.6B/glove.6B.50d.txt'):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = set()
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        model.add(word)

    print("Done.",len(model)," words loaded!")
    return model

## too slow here
def embed_to_word(embd,model):
    bestWord = None
    distance = float('inf')
    for word in model.keys():
        e=model[word]
        d = 0
        for a,b in zip(e,embd):
            d+=(a-b)*(a-b)
        if d<distance:
            distance=d
            bestWord = word

    assert(bestWord is not None)
    return (bestWord, distance)


def glove_example():
    #importing the glove library
    from glove import Corpus, Glove
    # creating a corpus object
    corpus = Corpus() 
    
    lines = [["hello", 'this','tutorial', 'on', 'how','convert' ,'word','integer','format'],['this','beautiful', 'day'],['Jack','going','office']]  
    #training the corpus to generate the co occurence matrix which is used in GloVe
    corpus.fit(lines, window=10)
    #creating a Glove object which will use the matrix created in the above lines to create embeddings
    #We can set the learning rate as it uses Gradient Descent and number of components
    glove = Glove(no_components=20, learning_rate=0.05)
    
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=False)
    glove.add_dictionary(corpus.dictionary)
    #glove.save('glove.model')
    vector = glove.word_vectors[glove.dictionary['hello']]
    print(vector)
    print(glove.dictionary['hello'])

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&()+,-./;~'

limit = {
        'maxq' : 25,
        'minq' : 2,
        'maxa' : 25,
        'mina' : 2
        }

UNK = 'unk'
VOCAB_SIZE = 30000


import random

import nltk
import itertools
from collections import defaultdict

import numpy as np

import pickle
from autocorrect import spell
import re   


def filter_line(line,whitelist):
    line = line.replace("-"," ")
    return ''.join([ch for ch in line if ch in whitelist])

def spell_check(sentences):
    for sen in sentences:
        for i in range(len(sen)):
            sen[i] = spell(sen[i])


'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )
'''
def filter_data(sequences):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences)//2

    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i+1].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i+1])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a

'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )
'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist

'''
 filter based on number of unknowns (words not in vocabulary)
  filter out the worst sentences
'''
def filter_unk(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)

    filtered_q, filtered_a = [], []

    for qline, aline in zip(qtokenized, atokenized):
        unk_count_q = len([ w for w in qline if w not in w2idx ])
        unk_count_a = len([ w for w in aline if w not in w2idx ])
        if unk_count_a <= 2:
            if unk_count_q > 0:
                if unk_count_q/len(qline) > 0.2:
                    pass
            filtered_q.append(qline)
            filtered_a.append(aline)

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((data_len - filt_data_len)*100/data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a

'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )
 
'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32) 
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]
'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))

from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string 
from pattern.en import suggest


def filter_sentence(line):
    line = filter_line(line,EN_WHITELIST)
    #line = line.lower()
    tokens = word_tokenize(line)
    # convert to lower case
    tokens = [w.lower() for w in tokens]

    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    words = [word for word in stripped if word.isalpha()]
    #porter = PorterStemmer()
    #stemmed = [porter.stem(word) for word in words]
    return words

def filter_sentence_2(words,glove_model):
    #words = filter_sentence(line)
    new_words = []
    for i in range(len(words)):
        if words[i] not in glove_model:
            #new_word = suggest(words[i])[0][0]
            #if words[i] in glove_model:
            new_word = simple_fuzzy_checking(words[i],glove_model)
            new_words.append(new_word)
        else:
            new_words.append(words[i])
    return new_words

def process_data(trainval_path='/HDD/dl_proj/msr_vtt/train_val_videodatainfo.json',test_path='/HDD/dl_proj/msr_vtt/test_videodatainfo.json'):
    
    # gloveFile='/HDD/dl_proj/glove/glove.6B.50d.txt'
    # print("Loading Glove Model")
    # f = open(gloveFile,'r')
    # glove_words = set()

    # for line in f:
    #     splitLine = line.split()
    #     word_ = splitLine[0]
    #     glove_words.add(word_)

    with open(trainval_path) as file:
        trainval_att = json.load(file)
    
    with open(test_path) as file:
        test_att = json.load(file)
    
    train_dict = dict()
    val_dict = dict()
    test_dict = dict()

    for item in trainval_att['videos']:
        if item['split'] == 'train':
            train_dict[item['video_id']] = {'category': item['category'],
                                            'caption':[],
                                            }
        else:
            val_dict[item['video_id']] = {'category': item['category'],
                                            'caption':[],
                                         }
    
    for item in test_att['videos']:
        test_dict[item['video_id']] = {'category': item['category'],
                                        'caption':[],
                                        }

    for item in trainval_att['sentences']:
        line = item['caption']
        line = filter_sentence(line)
        #line = filter_sentence_2(line,glove_words)
        if item['video_id'] in train_dict:
            train_dict[item['video_id']]['caption'].append(line)
        else:
            val_dict[item['video_id']]['caption'].append(line)
    
    for item in test_att['sentences']:
        line = item['caption']
        #line = filter_sentence_2(line,glove_words)
        line = filter_sentence(line)
        test_dict[item['video_id']]['caption'].append(line)

    # for video in train_dict:
    #     index = np.argmax([len(s) for s in train_dict[video]['caption']])
    #     words = train_dict[video]['caption'][index]
    #     words = filter_sentence_2(words,glove_words)
    #     train_dict[video]['caption'] = words
    
    # for video in val_dict:
    #     index = np.argmax([len(s) for s in val_dict[video]['caption']])
    #     words = val_dict[video]['caption'][index]
    #     words = filter_sentence_2(words,glove_words)
    #     val_dict[video]['caption'] = words
    
    # for video in test_dict:
    #     index = np.argmax([len(s) for s in test_dict[video]['caption']])
    #     words = test_dict[video]['caption'][index]
    #     words = filter_sentence_2(words,glove_words)
    #     test_dict[video]['caption'] = words

    
    """
    Save these dictionaries
    """
    with open('./annotation/train.json','w') as fp:
        json.dump(train_dict,fp)
    
    with open('./annotation/val.json','w') as fp:
        json.dump(val_dict,fp)
    
    with open('./annotation/test.json','w') as fp:
        json.dump(test_dict,fp)


    """
    Change the word 

    """
  

def new_annotation():
    with open('./annotation/train_50d.json','r') as fp:
        train_ = json.load(fp)
    
    with open('./annotation/test_50d.json','r') as fp:
        test_ = json.load(fp)
    
    with open('./annotation/val_50d.json','r') as fp:
        val_ = json.load(fp)
    
    print("Train length :",len(train_.keys()),"Test length :",len(test_.keys()), "Val length :",len(val_.keys()))
    length = dict()
    words = set()
    captions = []

    for item in train_:
        token = train_[item]['caption']
        
        if len(token) not in length:
            length[len(token)] = 1
        else:
            length[len(token)] += 1

        for word in token:
            words.add(word)
        
        # if len(token) < 4:
        #     print(token)


    for item in test_:
        token = test_[item]['caption']

        if len(token) not in length:
            length[len(token)] = 1
        else:
            length[len(token)] += 1

        for word in token:
            words.add(word)

        # if len(token) < 4:
        #     print(token)

    for item in val_:
        token = val_[item]['caption']
        
        if len(token) not in length:
            length[len(token)] = 1
        else:
            length[len(token)] += 1

        for word in token:
            words.add(word)
        
        # if len(token) < 4:
        #     print(token)
    
    print("Total Number of words :",len(words))

    total = np.sum(np.array(list(length.values())))
    print("Total :",total)

    len_ = 10
    i = 0
    for key in length:
        print(key,length[key])
        if key < len_:
            i += length[key]
    
    print("Length < ",len_,":",i)

from fuzzywuzzy import fuzz

def simple_fuzzy_checking(word,glove_model):
    #glove_model = loadGloveModel_2()
    ratio = 0
    simliar_word = None
    for _ in glove_model:
        r = fuzz.ratio(word,_)
        if r > ratio:
            ratio = r
            simliar_word = _

    return simliar_word

def main():
    print("--")
    #read_json()
    #process_data(trainval_path='/home/jiaming/Downloads/dataset/msr-vtt/train_val_annotation/train_val_videodatainfo.json',test_path='/home/jiaming/Downloads/dataset/msr-vtt/test_videodatainfo.json')
    new_annotation()
    # path = '/HDD/dl_proj/glove/glove.6B.50d.txt'
    # glove_model = loadGloveModel_2(path)
    # word = 'redblue'
    # best = simple_fuzzy_checking(word,glove_model)
    #print(best)

    

if __name__ == '__main__':
    main()