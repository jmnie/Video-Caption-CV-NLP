import cv2
import mxnet.gluon.data as data
import json
import os
import numpy as np

def make_dataset(rootdir,dict_file):
    with open(dict_file,'r') as fp:
        video_dict = json.load(fp)
    videos = []
    for key in video_dict:
        key = key + '.mp4'
        path = os.path.join(rootdir,key)
        caption = video_dict[key]['caption']
        item = (path,caption)
        videos.append(item)

    return videos

def load_glove_model(gloveFile):
    f = open(gloveFile,'r')
    model = {}
    dimension = 0
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
        dimension = len(embedding)
    return model,dimension


def word2embd(words,glove_model,caption_length,dimension):
    embd = np.zeros((caption_length,dimension))
    for i in range(caption_length):
        if i < len(words):
            embd[i] = glove_model[words[i]]
    return embd

def opencv_loader(path,frame_count):
    videocap = cv2.VideoCapture(path)
    count = 0
    frames = []

    while True:
        ret, frame = videocap.read()
        frames.append(frame)
        count += 1
        if count == frame_count:
            break

    return frame


class videoFolder(data.Dataset):
    def __init__(self, rootdir, dict_file, img_size, frames, gloveFile, caption_length, transform=None,target_transform=None, loader=opencv_loader, word2embd=word2embd):

        videos = make_dataset(rootdir,dict_file)

        if len(videos) == 0:
            raise(RuntimeError("Found 0 videos in directory of: " + rootdir + "\n"))
        
        self.videos = videos

        self.loader = loader
        self.word2embd = word2embd

        self.rootdir = rootdir
        self.dict_file = dict_file
        self.transform = transform
        self.img_size = img_size
        self.frames = frames
        glove_model,dimension = load_glove_model(gloveFile)
        self.glove_model = glove_model
        self.dimension = dimension
        self.caption_length = caption_length
       
        #return super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, caption = self.videos[idx]
        video_frame = self.loader(path)
        embd = self.word2embd(caption,self.glove_model,self.caption_length,self.dimension)

        """
        Preprocess the frame
        """
        video_frame = self.transform(video_frame)
        
        return video_frame,embd
