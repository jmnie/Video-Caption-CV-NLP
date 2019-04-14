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

EN_BLACKLIST = '!#$%&()*+,-./:;<=>?@[]^_`{|}~'

def read_json(path='/home/jiaming/Downloads/dataset/msr-vtt/test_videodatainfo.json'):
    with open(path) as file:
        attribute = json.load(file)
    
    #print(attribute.keys())
    #print(attribute['info'].keys())
    #print(attribute['info'].values())
    #print(attribute['videos'][0])
    #print(attribute['sentences'][0])

    EN_BLACKLIST = '!#$%&()*+,-./:;<=>?@[]^_`{|}~'
    vocab = set()
    caption_length = dict()
    for item in attribute['sentences']:
        #item = re.sub(r'!#$%&(),-./:;=@^_`{}~', '', item)
        caption = item['caption']#.split()
        caption = re.sub(r'!#$%&(),-./:;=@^_`{}~', '', caption)
        caption = caption.split()
        #print(caption)

        if len(caption) in caption_length:
            caption_length[len(caption)] += 1
        else:
            caption_length[len(caption)] = 1

        for word in caption:
            #word = re.sub(r'!#$%&(),-./:;=@^_`{}~', '', word)
            vocab.add(word)

    #for key in caption_length:
    #    print(key,caption_length[key])

    return vocab

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
    

def main():
    #res_1 = read_json(path='/home/jiaming/Downloads/dataset/msr-vtt/test_videodatainfo.json')
    print("--")
    #res_2 = read_json('/home/jiaming/Downloads/dataset/msr-vtt/train_val_annotation/train_val_videodatainfo.json')
    #res = res_1 | res_2
    #print(len(res))

    from nltk import sent_tokenize
    test = "an ad for a cooking show (home aux fourne aux) is shown while the camera pans to various video clips of someone preparing what appears to be tarts in a kitchen"
    test = sent_tokenize(test)
    print(test[0])

    #model = loadGloveModel()
    # print("Loading Glove Model")
    # f = open('/home/jiaming/Downloads/dataset/glove.840B.300d.txt','r')
    # #model = {}

    # new_word = set()

    # for line in f:
    #     splitLine = line.split()
    #     word = splitLine[0]
    #     new_word.add(word)
    #     #embedding = np.array([float(val) for val in splitLine[1:]])
    #     #model[word] = embedding

    # i = 0
    # for word in res:
    #     if word not in new_word:
    #         i += 1
    #         print(word)
    # print(i)
    #test_vector = np.random.uniform(0,0,size=(50))
    #print(test_vector)
    #print(test_vector.shape)
    #res = embed_to_word(test_vector,model)
    #print(res[0],res[1])
    # print(len(res_1.keys()))
    # for word in res_1:
    #     if res_1[word] > 20:
    #         print(word,res_1[word])

if __name__ == '__main__':
    main()